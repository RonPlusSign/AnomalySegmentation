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
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from enet import ENet
from bisenet import BiSeNetV1
import sys
seed = 42

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

def mahalanobis_distance(f_x, mean, covariance_inv):
    """
    Compute the Mahalanobis distance between a sample f_x and a class mean with the inverse covariance matrix.
    f_x: test sample feature vector (numpy array of shape (20, 512, 1024))
    mean: class mean (vector of shape (20))
    covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
    Returns the Mahalanobis distance (scalar)
    """
    print(f"mean shape: {mean.shape}")
    print(f"covariance_inv shape: {covariance_inv.shape}")
    print(f"f_x shape: {f_x.shape}")
    diff = f_x.reshape(20, -1) - mean # Reshape to (20, 512*1024) because covariance is 20x20
    return diff.T @ (covariance_inv @ diff) # FIXME: This allocates too much memory and doesn't work

def mahalanobis_score(f_x, means, covariance_inv):
    """
    Compute the Mahalanobis distance-based confidence score for a test sample.
    
    f_x: test sample feature vector (numpy array of shape (20, 512, 1024))
    means: class means (numpy array of shape (20))
    covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
    Returns the confidence score (scalar)
    """
    max_distance = float('-inf')  # Start with a very small number
    for c in range(means.shape[0]):  # Loop over each class
        distance = mahalanobis_distance(f_x, means, covariance_inv)
        print(f"Distance for class {c}: {distance}") 
        score = -distance
        if score > max_distance:
            max_distance = score
    return max_distance

def mahalanobis_distance_score(output, centers, inv_cov_matrix, layer=16):
    """
    Computes Mahalanobis distances of a batch, when the centers and covariance
    matrix are already computed, without preprocessing.
    """
    num_classes = len(centers)
    zero_m_feat = output - centers[: , None, None ] #[output - centers[c] for c in range(num_classes)]
    #zero_m_feat = torch.stack(zero_m_feat)
    distances = -torch.matmul(zero_m_feat, inv_cov_matrix).matmul(zero_m_feat.transpose(1,2)).diagonal()
    return(distances.max(1).values)

# def mahalanobis_distance_per_pixel(f_x_pixel, mean, covariance_inv):
#     """
#     Compute the Mahalanobis distance for a single pixel's feature vector.
#     f_x_pixel: feature vector for a single pixel (numpy array of shape (20,))
#     mean: class mean (numpy array of shape (20,))
#     covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
#     Returns the Mahalanobis distance (scalar).
#     """
#     diff = f_x_pixel - mean  # Difference between pixel feature and class mean
#     return diff.T @ covariance_inv @ diff  # Mahalanobis distance


# def mahalanobis_score(f_x, means, covariance_inv):
#     """
#     Compute the Mahalanobis distance-based confidence score for a test sample (image).
    
#     f_x: test sample feature vector (numpy array of shape (20, 512, 1024))
#     means: class means (numpy array of shape (20, 20)) - one mean vector per class
#     covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
#     Returns the confidence score (scalar).
#     """
#     C, H, W = f_x.shape[0], f_x.shape[1], f_x.shape[2]  # Classes, height, width
#     max_scores = np.full((H, W), float('-inf'))  # Max scores for each pixel

#     for c in range(means.shape[0]):  # Loop over each class (C = 20)
#         mean = means[c]  # Mean vector for class c (shape: (20,))
#         for h in range(H):
#             for w in range(W):
#                 f_x_pixel = f_x[:, h, w]  # Feature vector for a single pixel (shape: (20,))
#                 distance = mahalanobis_distance_per_pixel(f_x_pixel, mean, covariance_inv)
#                 score = -distance  # Convert distance to confidence score
#                 if score > max_scores[h, w]:  # Keep the max score for each pixel
#                     max_scores[h, w] = score

#     # Aggregate the scores over all pixels (e.g., average, sum, or max)
#     return np.mean(max_scores)  # Return the mean confidence score


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/Validation_Dataset/RoadObsticle21/images/*.webp",
        help="A single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/AnomalySegmentation/trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method',default="MSP") #can be MSP or MaxLogit or MaxEntropy or Mahalanobis
    parser.add_argument('--void', action='store_true')
    
    parser.add_argument('--temperature', default=0) # add the path of the model absolute path
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    method = args.method

    if not os.path.exists(f'results-{method}.txt'):
        open(f'results-{method}.txt', 'w').close()
    file = open(f'results-{method}.txt', 'a')

    modelpath = args.loadDir +"/" +args.model + ".py"
    weightspath = args.loadDir + args.loadWeights

    temperature = float(args.temperature)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    if args.model == "erfnet":
        model = ERFNet(NUM_CLASSES).to(device)
    elif args.model =="enet":
        model = ENet(NUM_CLASSES).to(device)
    elif args.model == "bisenet":
        model = BiSeNetV1(NUM_CLASSES).to(device)

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
    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))

    if(temperature != 0 ):
        model = ModelWithTemperature(model, temperature = temperature)

    print ("Model and weights LOADED successfully")
    model.eval()

    if method == "Mahalanobis":
        # Load mean and covariance matrices from "save" folder
        means = np.load("/content/AnomalySegmentation/save/mean_cityscapes_erfnet_20_elements.npy")
        cov = np.load("/content/AnomalySegmentation/save/cov_matrix_erfnet_20x20.npy")
        cov_inv = np.linalg.inv(cov)
        print("mean shape: ", means.shape)
        print("cov shape: ", cov.shape)
        
        # Convert to PyTorch tensors
        means = torch.from_numpy(means).to(device)
        cov_inv = torch.from_numpy(cov_inv).to(device)
    
    for path in glob.glob(os.path.expanduser(str(args.input))):
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            if args.model == "bisenet":
                result = model(images)[0].squeeze(0)
            else: #TODO check ENet output
                result = model(images).squeeze(0)

        if args.void:
            anomaly_result = F.softmax(result, dim=0)[-1]
            anomaly_result = anomaly_result.data.cpu().numpy()
        else:
            result_no_void = result[:-1] # remove the last channel (void)
            
            if(method == "MaxLogit"):
                anomaly_result = -torch.max(result_no_void, dim=0)[0]
                anomaly_result = anomaly_result.data.cpu().numpy()
            elif(method == "MaxEntropy"):
                probs = F.softmax(result_no_void, dim=0)
                entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=0), torch.log(torch.tensor(probs.shape[0])))
                anomaly_result = entropy.data.cpu().numpy().astype("float32")
            elif(method == "MSP"):
                anomaly_result = 1.0 - torch.max(F.softmax(result_no_void, dim=0), dim=0)[0]
                anomaly_result = anomaly_result.data.cpu().numpy()
            elif(method == "Mahalanobis"):
                # Compute Mahalanobis distance
                anomaly_result = mahalanobis_distance_score(F.softmax(result, dim=0), means, cov_inv)
                anomaly_result = anomaly_result.data.cpu().numpy()
                print(f"Mahalanobis score: {anomaly_result}")
            else:
                raise ValueError("Invalid method")

        # Load ground truth mask
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        print(f"Loaded out-of-distribution ground-truths: {ood_gts}") #debug
        
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts) #ho verificato ci sono veramente dei 2 all'interno dell'immagine
        if "LostAndFound" in pathGT: # LostAndFound qui non entra
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT: #  qui non entra
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)

        del result, anomaly_result, ood_gts, mask, image
        torch.cuda.empty_cache()

    file.write( "\n")

    # Calculate metrics: AUPRC and FPR@TPR95
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

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()