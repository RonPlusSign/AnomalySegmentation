# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

from icecream import ic
from temperature_scaling import ModelWithTemperature
import torch.nn.functional as F # aggiunto io 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize #aggiunto io
import sys
seed = 42

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
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method',default="MSP") #can be MSP or MaxLogit or MaxEntropy
    
    parser.add_argument('--temperature', default=0) # add the path of the model absolute path
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    method = args.method

    if not os.path.exists(f'results-{method}.txt'):
        open(f'results-{method}.txt', 'w').close()
    file = open(f'results-{method}.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    temperature = float(args.temperature)
    

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
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

    if(temperature != 0 ):
        model = ModelWithTemperature(model, temperature = temperature)

    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        print(f"result.shpe {result.shape}")#debug
        probabilities = F.softmax(result, dim=1)
        print(f"result.squeeze(0).data.cpu().numpy() : { probabilities.squeeze(0).data.cpu().numpy().sum() }") #debug
        
        if(method == "MaxLogit"):
            anomaly_result = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)   
        elif(method == "MaxEntropy"):
            # da sistemare non il massimo
            def get_softmax(network, image, transform=None, as_numpy=True):
                if transform is None:
                    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                #x = transform(image)
                x = ToTensor()(image)
                if (not args.cpu):
                    x = x.unsqueeze_(0).cuda()
                else : 
                    x = x.unsqueeze_(0)
                    
                with torch.no_grad():
                    y = network(x)
                probs = F.softmax(y, 1)
                if as_numpy:
                    probs = probs.data.cpu().numpy()[0].astype("float32")
                return probs


            def get_entropy(network, image, transform=None, as_numpy=True):
                probs = get_softmax(network, image, transform, as_numpy=False)
                entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=1), torch.log(torch.tensor(probs.shape[1])))
                if as_numpy:
                    entropy = entropy.data.cpu().numpy()[0].astype("float32")
                return entropy

            anomaly_result = get_entropy(model, Image.open(path).convert('RGB'))
        else :#MSP
            anomaly_result = 1.0 - np.max(probabilities.squeeze(0).data.cpu().numpy(), axis=0)
            #anomaly_result = 1.0 - torch.max(F.softmax(result / args.temperature, dim=0), dim=0)[0]
            ic(result)
            ic(anomaly_result)
            
            #anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0) com'era prima MSP             
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)
        print(f"ood_gts appena caricato : {ood_gts}") #debug
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts) #ho verificato ci sono veramente dei 2 all'interno dell'immagine
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list) 
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1) #array con True dove ood_gts == 1 resto false
    ind_mask = (ood_gts == 0) #array con True dove ood_gts == 0 resto false

    ood_out = anomaly_scores[ood_mask] # prendo soltanto i punteggi degli ood
    ind_out = anomaly_scores[ind_mask] # prendo soltanto i punteggi degli ind

    ood_label = np.ones(len(ood_out)) # creo un array di 1 farà da label
    ind_label = np.zeros(len(ind_out)) # creo un array di 0 farà da label
    
    val_out = np.concatenate((ind_out, ood_out)) # array con tutti i punteggi delle classi ind e out
    val_label = np.concatenate((ind_label, ood_label)) #array con tutte le label delle classi ind e out

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    #Mean Intersection Over Union
    threshold = 0.5
    
    #ood class
    val_pred = (val_out >= threshold).astype(int)

    # Calcolo IoU
    tp = np.sum((val_pred == 1) & (val_label == 1))
    fp = np.sum((val_pred == 1) & (val_label == 0))
    fn = np.sum((val_pred == 0) & (val_label == 1))
    tn = np.sum((val_pred == 0) & (val_label == 0))

    iou_ood = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    iou_ind = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0

    # Media degli IoU
    mIoU = (iou_ood + iou_ind) / 2
    print(f'mIoU: {mIoU*100.0}') # da ricontrollare 

    #print(f'val_out : {val_out} \n val_label : {val_label}')

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()