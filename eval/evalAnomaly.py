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
import torch.nn.functional as F 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from enet import ENet
from bisenet import BiSeNetV1
import sys
from tqdm import tqdm
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

def mahalanobis_distance_score(output, means, cov_inv):
    """
    Compute the Mahalanobis distance-based confidence score for a test sample.
    
    f_x: test sample feature vector (numpy array of shape (20, 512, 1024))
    means: class means (numpy array of shape (20,20))
    covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
    Returns the confidence score (512x1024)
    """
    """
    # Inizializza il risultato finale per i punteggi
    M_scores = torch.empty(512, 1024)  # Uno score per ogni pixel

    # Itera sui pixel
    for i in range(512):  # Altezza
        for j in range(1024):  # Larghezza
            # f(x) per il pixel corrente
            f_x = output[:, i, j]  # (20,)

            # Calcola lo score per ciascuna classe c
            scores = []
            for c in range(NUM_CLASSES):  # Numero di classi
                mean_c = means[c]  # Media per la classe c
                centered = f_x - mean_c  # (20,)
                score_c = -torch.matmul(centered.T, torch.matmul(cov_inv, centered))  # Scala
                scores.append(score_c)

            # Trova il massimo tra tutti gli score
            M_scores[i, j] = max(scores)

    return M_scores
    """
    """
    # Inizializza il risultato finale per i punteggi
    M_scores = torch.empty(512, 1024, device=output.device)  # Uno score per ogni pixel

    # Reshape output per avere i pixel come dimensione principale
    output_reshaped = output.permute(1, 2, 0).reshape(-1, output.size(0))  # (512*1024, 20)

    # Calcola lo score per ciascuna classe c
    scores = torch.empty((512 * 1024, NUM_CLASSES), device=output.device)
    for c in range(NUM_CLASSES):
        mean_c = means[c]  # Media per la classe c
        centered = output_reshaped - mean_c  # (512*1024, 20)
        scores[:, c] = -torch.einsum('ij,jk,ik->i', centered, cov_inv, centered)  # (512*1024,)

    # Trova il massimo tra tutti gli score
    M_scores = scores.max(dim=1)[0].reshape(512, 1024)

    print(f"Mahalanobis score: {M_scores.shape}")

    return M_scores
    """

    """
    # Dimensioni input
    NUM_CLASSES = 20

    # output: (20, 512, 1024) -> le feature per ogni pixel
    # means: (20, 20) -> le medie per ciascuna classe
    # cov_inv: (20, 20) -> matrice di covarianza inversa

    # Reshape delle medie per il broadcasting
    means = means.view(NUM_CLASSES, NUM_CLASSES, 1, 1)  # (20, 20, 1, 1)

    # Calcolo di f(x) - μ_c per tutte le classi e tutti i pixel
    centered = output.unsqueeze(1) - means  # (20, 20, 512, 1024)

    # Applica la matrice di covarianza inversa
    cov_centered = torch.einsum('ab,bcde->acde', cov_inv, centered)  # (20, 20, 512, 1024)

    # Prodotto scalare per il termine quadratico
    quad_form = torch.einsum('bcde,bcde->bde', centered, cov_centered)  # (20, 512, 1024)

    # Trova il massimo score per ogni pixel
    M_scores = -quad_form.max(dim=0).values  # (512, 1024)

    return M_scores
    """
    num_features, height, width = output.shape

    # Trasponi e rimodella l'output per gestire i pixel in batch
    output_flat = output.view(num_features, -1).T  # (height * width, num_features)

    # Espandi le medie per il calcolo batch
    means_expanded = means.unsqueeze(1)  # (num_classes, 1, num_features)
    output_expanded = output_flat.unsqueeze(0)  # (1, height * width, num_features)

    # Calcola la differenza (broadcasting)
    centered = output_expanded - means_expanded  # (num_classes, height * width, num_features)

    # Calcola lo score di Mahalanobis
    scores = -torch.einsum(  
        'npi,ij,npj->np',
        centered, cov_inv, centered
    )  # (num_classes, height * width)

    # Trova il massimo score per ogni pixel
    M_scores_flat = torch.abs(scores.max(dim=0).values)  #TODO  (height * width) era senza abs
    # Rimodella in (height, width)
    M_scores = M_scores_flat.view(height, width)

    # Verifica un pixel specifico
    i, j = 100, 200
    f_x_manual = output[:, i, j]
    scores_manual = []
    for c in range(NUM_CLASSES):
        mean_c = means[c]
        centered = f_x_manual - mean_c
        score_c = -torch.matmul(centered.T, torch.matmul(cov_inv, centered))
        scores_manual.append(score_c)
    manual_max = max(scores_manual)

    assert torch.isclose(M_scores[i, j], manual_max), "L'ordine non è mantenuto!"
    print("Ordine verificato e corretto!")

    return M_scores



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
        means = np.load("/content/AnomalySegmentation/save/mean_cityscapes_erfnet.npy")
        cov = np.load("/content/AnomalySegmentation/save/cov_cityscapes_erfnet.npy")
        cov_inv = np.linalg.inv(cov)
        print("mean shape: ", means.shape)
        print("cov shape: ", cov.shape)
        
        # Convert to PyTorch tensors
        means = torch.from_numpy(means).to(device)
        cov_inv = torch.from_numpy(cov_inv).to(device)
    
    for path in tqdm(glob.glob(os.path.expanduser(str(args.input)))):
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
                anomaly_result = mahalanobis_distance_score(result, means, cov_inv)
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

        del result, anomaly_result, ood_gts, mask, images
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