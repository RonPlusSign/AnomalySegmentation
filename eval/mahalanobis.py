# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import random
from PIL import Image, ImageOps
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from transform import Relabel, ToLabel, Colorize
from enet import ENet
from bisenet import BiSeNetV1
from dataset1 import cityscapes
from torch.utils.data import DataLoader
from tqdm import tqdm

seed = 42

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/Validation_Dataset/RoadObsticle21/images/*.webp",
        help="A single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/AnomalySegmentation") 
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default="erfnet") 
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--mean', default = '') #/save/mean_cityscapes_erfnet.npy
    args = parser.parse_args()

    #modelpath = args.loadDir +"/" +args.model + ".py"
    weightspath = args.loadDir + args.loadWeights
    mean_is_computed = len(args.mean) > 0
    mean_path = args.loadDir + args.mean

    print ("Loading model: " + args.model)
    print ("Loading weights: " + weightspath)
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    if mean_is_computed:
        pre_computed_mean = np.load(mean_path)
        pre_computed_mean = torch.from_numpy(pre_computed_mean).cuda()
        print(f"pre_computed_mean {pre_computed_mean.shape}")
    
    # Augmentations and Normalizations
    co_transform = MyCoTransform(False, augment=False, height=512)#1024)
    
    # Dataset and Loader
    dataset_train = cityscapes(args.datadir, co_transform, 'train')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    if args.model == "erfnet":
        model = ERFNet(NUM_CLASSES)
    elif args.model =="enet":
        model = ENet(NUM_CLASSES)
    elif args.model == "bisenet":
        model = BiSeNetV1(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        print(state_dict.keys())
        print(own_state.keys())
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    

    # Covariance matrices
    cov_matrix = torch.zeros((20, 20), dtype=torch.float32, device='cuda')
    num_images = 0  

    sum_per_class = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    pixel_count_per_class = np.zeros((NUM_CLASSES,), dtype=np.int32)

    for images, labels in tqdm(loader):
        if not args.cpu:
            images = images.cuda()
            #labels = labels.cuda()
           
        output = None
        with torch.no_grad():
            if args.model == "bisenet":
                result = F.softmax(model(images)[0].squeeze(0), dim=0)
                output = result.data.cpu().numpy()
            else: #TODO check ENet output
                result = F.softmax(model(images).squeeze(0), dim=0)
                output = result.data.cpu().numpy()
            result = result.reshape(20, -1) # Convert from (20, 512, 1024) to (20, 512*1024)
        
        # If mean is not computed, accumulate sum and count per class
        if not mean_is_computed:
            # Accumulate the sum of the output for each class
            for b in range(args.batch_size):
                for c in range(NUM_CLASSES):
                    # Create a mask for the pixels corresponding to class `c`
                    mask = (labels[b] == c)
                    print(f"Mask shape: {mask.shape}")
                    print(f"Output shape: {output.shape}")
                    print(f"output[b, :, mask] shape: {output[b, :, mask].shape}")
                    
                    # Accumulate the sum of the output for class `c`
                    sum_per_class[c] += np.sum(output[b, :, mask])
                    print(f"Sum per class shape: {sum_per_class[c].shape}")

                    # Accumulate the count of pixels for class `c`
                    pixel_count_per_class[c] += np.sum(mask)

        else:
            centered = result - pre_computed_mean.reshape(-1, 1) # Convert from (20,) to (20, 1) to allow broadcasting
            cov_matrix += centered @ centered.T
            # for c in range(NUM_CLASSES):
            #     # Center the output relative to the precomputed mean
            #     centered = result[c] - pre_computed_mean[c]
            #     cov_matrix += centered.T @ centered
        num_images += images.size(0)

    # After processing all images, calculate the mean per class
    if not mean_is_computed:
        mean = sum_dataset / (num_images * 512 * 1024) # Normalize by the number of pixels
        
        # for c in range(NUM_CLASSES):
            # if pixel_count_per_class[c] > 0:
            #     # Divide the sum for class 'c' by the count of pixels for class 'c'
            #     #mean[c] = sum_dataset[c] / pixel_count_per_class[c]
            #     mean[c] = sum_dataset[c] / num_images
        
        print(f"Mean per class: {mean.shape}")
        np.save(f"{args.loadDir}/save/mean_cityscapes_{args.model}.npy", mean)
        print(f"Mean output saved as '{args.loadDir}/save/mean_cityscapes_{args.model}.npy'")
    else: 
        cov_matrix /= (512 * 1024 * num_images) # Normalize by the number of pixels
        print(f"Covariance matrix: {cov_matrix.shape}")
        np.save(f"{args.loadDir}/save/cov_matrix_{args.model}.npy", cov_matrix.data.cpu().numpy())
        print(f"Covariance matrice saved as '{args.loadDir}/save/cov_matrix_{args.model}.npy'")
 

if __name__ == '__main__':
    main()