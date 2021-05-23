import torch
import numpy as np
from torch.nn import Module
import torch.nn.functional as F


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, beta=0.8, reduce=False):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y, ind, record):
        # cross_entropy = - t * log(p)
        noise_confidence = np.array([np.mean(record[ind[i]]) for i in range(len(ind))])
        beta_confidence = noise_confidence/np.sum(noise_confidence)*self.beta
        # noise_confidence = noise_confidence/np.sum(noise_confidence)
        # beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')
        beta_xentropy = F.cross_entropy(y_pred, y, reduce=False)*torch.from_numpy(beta_confidence).to('cuda').float()
        beta_xentropy = torch.sum(beta_xentropy)
        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        bootstrap_confidence = (1-noise_confidence)/np.sum(1-noise_confidence)*(1-self.beta)
        bootstrap = bootstrap*torch.from_numpy(bootstrap_confidence).to('cuda').float()
        bootstrap = -torch.sum(bootstrap)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap