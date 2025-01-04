import torch.nn as nn
import torch.nn.functional as F
import torch

# Code from: https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/master/losses/isomaxplus.py
# Paper: https://arxiv.org/pdf/2105.14399

class IsoMaxPlusLossFirstPart(nn.Module):
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        """
        features: B x C x H x W
        Returns logits: B x num_classes x H x W
        """
        B, C, H, W = features.size()  # Batch, Channels, Height, Width
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # Reshape to (B*H*W) x C
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0
        )
        logits = -distances  # Negative distances as logits
        logits = logits.view(B, H, W, self.num_classes).permute(0, 3, 1, 2)  # Reshape to B x num_classes x H x W
        return logits / self.temperature
    
class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, inter_distances