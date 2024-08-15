import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            
            loss = self.alpha[targets] * loss
        return loss.mean()
    
def init_loss(loss_fn, focalLoss_gamma, loss_weights = None, device="cpu"):
    if loss_weights is not None:
        loss_weights = torch.tensor(loss_weights, dtype=torch.float).to(device)
        
    if loss_fn == "focal":
        criterion = FocalLoss(alpha = loss_weights ,gamma=focalLoss_gamma, device = device)

    elif loss_fn == "CCE":
        
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    return criterion