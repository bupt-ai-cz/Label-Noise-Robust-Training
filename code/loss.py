import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

# Loss functions
def loss_coweight(y_1, y_2, t, epoch, forget_rate, ind, noise_or_not, record_1, record_2):
    # m_prob_1 = np.array([1-(np.mean(record_1[ind[i]])) for i in range(len(ind))])
    # m_prob_2 = np.array([1-(np.mean(record_2[ind[i]])) for i in range(len(ind))])
    # weight_1 = m_prob_2/np.sum(m_prob_2)
    # weight_2 = m_prob_1/np.sum(m_prob_1)
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    # ind_1_sorted = np.argsort(m_prob_1)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    # # ind_2_sorted = np.argsort(m_prob_2)
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    # remember_rate2 = 1 - forget_rate-0.05
    # num_remember2 = int(remember_rate2 * len(loss_2_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)
   
    prob_1 = torch.gather(F.softmax(y_1, dim=1),1,t.view(t.size()[0],1))
    prob_1 = prob_1.view(t.size()[0])
    prob_2 = torch.gather(F.softmax(y_2, dim=1),1,t.view(t.size()[0],1))
    prob_2 = prob_2.view(t.size()[0])
    
    # weight_1 = loss_1[ind_2_update]/torch.sum(loss_1[ind_2_update])
    # weight_2 = loss_2[ind_1_update]/torch.sum(loss_2[ind_1_update])
    # if epoch<5:
    #     loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduce = True)
    #     loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduce = True)
    # else:
        # ind_1_update=ind_1_sorted[:num_remember2]
    weight_1 =(torch.pow(1-prob_1,2)/torch.sum(torch.pow(1-prob_2,2)))
    weight_2 = torch.pow(1-prob_2,2)/torch.sum(torch.pow(1-prob_1,2))
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduce = False)*weight_1[ind_2_update]
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduce = False)*weight_2[ind_1_update]

    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduce = False)*torch.from_numpy(weight_1).to('cuda').float()
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduce = False)*torch.from_numpy(weight_2).to('cuda').float()
    return torch.sum(loss_1_update), torch.sum(loss_2_update), pure_ratio_1, pure_ratio_2


def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update), torch.sum(loss_2_update), pure_ratio_1, pure_ratio_2

def make_one_hot(label, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         label: A tensor of shape [N, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    label = label.reshape(label.shape[0],1)
    result = torch.zeros(label.shape[0],num_classes).cuda()
    result = result.scatter_(1, label, 1)
    return result
    
class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=2:
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = make_one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss