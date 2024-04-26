import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


################################################## BRUKES IKKE ###############################################################
class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        dice_loss = self.dice_loss(outputs, labels)
        cross_entropy_loss = self.cross_entropy_loss(outputs, labels)
        return dice_loss + cross_entropy_loss  # Combine the losses

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        probs = torch.softmax(outputs, dim=1)
        labels = labels.squeeze(1).long()
        true_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        true_one_hot = true_one_hot.permute(0, 4, 1, 2, 3)  # Adjust dimensions to match output

        dims = (0,) + tuple(range(2, outputs.ndimension()))
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)

        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = 1 - dice_score
        return dice_loss.mean()

# When initializing CombinedLoss, specify the number of classes
num_classes = 2  ##2 unique labels
loss_func = CombinedLoss(num_classes)

