# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image, ImageOps
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

from icecream import ic
from temperature_scaling import ModelWithTemperature
import torch.nn.functional as F # aggiunto io 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from transform import Relabel, ToLabel, Colorize
from enet import ENet
from bisenet import BiSeNetV1
import sys
seed = 42


from dataset1 import cityscapes
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
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


input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        #  Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

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
    anomaly_score_list = []
    ood_gts_list = []

 
    #modelpath = args.loadDir +"/" +args.model + ".py"
    weightspath = args.loadDir + args.loadWeights
    mean_is_computed = len(args.mean) > 0
    mean_path = args.loadDir + args.mean


    print ("Loading model: " + args.model)
    print ("Loading weights: " + weightspath)
    
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    if mean_is_computed :
        pre_computed_mean = np.load(mean_path)
        print(f"pre_computed_mean {pre_computed_mean.shape}")
    # Augmentations and Normalizations
    co_transform = MyCoTransform(False, augment=False, height=512)#1024)
    co_transform_val = MyCoTransform(False, augment=False, height=512)#1024)

    # Dataset and Loader
    dataset_train = cityscapes(args.datadir, co_transform, 'train')#senza co_transform non funziona perché non lo trasforma in tensore
    #dataset_train = cityscapes(args.datadir, co_transform, 'train')
    #dataset_val = cityscapes(args.datadir, co_transform_val, 'val') serve solo train

    # Calcoliamo i pesi delle classi dal dataset di addestramento
    #weights = calculate_class_weights(dataset_train, NUM_CLASSES)
    #print("Pesi delle classi:", weights)

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    #loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False) serve solo train 
    

    
    
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

    
    sum_dataset = np.zeros((20, 512, 1024), dtype=np.float32)
    if mean_is_computed:
        num_classes, height, width = pre_computed_mean.shape
        cov_matrices = np.zeros((num_classes, height * width, height * width), dtype=np.float32)

    num_images = 0 
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for step, (images, labels) in enumerate(loader):
        print(f"-----------{step/2974 * 100}-----------")
        #images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        #images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        if not args.cpu:
                images = images.cuda()
                #labels = labels.cuda()
        #print(f"images {images.shape}")
        #print(f"labels {labels.shape}")
       
            
        output = None
        with torch.no_grad():
            if args.model == "bisenet":
                result = model(images)[0].squeeze(0)
                output = result.data.cpu().numpy()
            else: #TODO check ENet output
                result = model(images).squeeze(0)
                output = result.data.cpu().numpy()
                #print(f"output {output.shape}")
        if not mean_is_computed : 
            sum_dataset += output
            
        else : # calcolo la covarianza
            for cls in range(num_classes):
                # Centrare rispetto alla media della classe
                centered = output[cls] - pre_computed_mean[cls]  # Forma (H, W)

                # Appiattire localmente (H x W)
                centered_flattened = centered.flatten()

                # Accumulare il prodotto centrato
                cov_matrices[cls] += np.outer(centered_flattened, centered_flattened) 
                
        num_images +=1
                    
    if not mean_is_computed:
        print(f"sum_dataset : {sum_dataset.shape}")
        print(f"num_images  : {num_images}")
        mean = sum_dataset / num_images
        print(f"mean : {mean.shape}")
        np.save(f"{args.loadDir}/save/mean_cityscapes_{args.model}.npy", mean)
        print(f"Mean output saved as '{args.loadDir}/save/mean_cityscapes_{args.model}.npy'")
    else : 
        # Normalizza ogni matrice di covarianza
        cov_matrices /= num_images
        print(f"cov_matrices : {cov_matrices.shape}")
        # Salva le matrici di covarianza per ogni classe
        np.save(f"{args.loadDir}/save/cov_matrices_{args.model}.npy", cov_matrices)
        print(f"Covariance matrices saved as '{args.loadDir}/save/cov_matrices_{args.model}.npy'")
        
 

if __name__ == '__main__':
    main()