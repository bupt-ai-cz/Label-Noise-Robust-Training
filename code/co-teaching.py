# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import myDataset
from sklearn.utils import shuffle
import argparse, sys
import numpy as np
import datetime
import shutil
import json
import pickle
import torchvision
import pandas as pd
import random
from loss import *

os.environ["CUDA_VISIBLE_DEVICES"]='0'
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'result/')
parser.add_argument('--data_dir', type = str, help = 'train file', default = '../../data/digest')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'digest, iciar, or colon', default = 'digest')
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=20)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    help="Use pre-trained models from the modelzoo",
    action="store_true",)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 56
learning_rate = args.lr 
filter_result = pd.read_csv('iteration/bishe/corrected_digest_40.csv')
train_data = pickle.load(open('pickle_data/digest_40.p', "rb"))

def record_history(index,output,target,recorder):
    prob = torch.gather(F.softmax(output, dim=1),1,target.view(target.size()[0],1))
    prob = prob.view(target.size()[0])
    for i,ind in enumerate(index):
        recorder[ind].append(prob[i].cpu().data.numpy().sum())

def meanStr(line):
    num_list = json.loads(line['probs'])
    mean = np.mean(np.array(num_list))
    return mean

# train_idx = len(train_data['trainIm'])
train_idx = len(filter_result)
y_train = filter_result['y'].values
y_train_noise = filter_result['correct_label'].values
# y_train = np.array(train_data['trainClass'])
# y_train_noise = np.array(train_data['trainNoiseClass'])
noise_or_not = (y_train==y_train_noise)
print(train_idx, np.sum(noise_or_not))
# data_dict['trainIm'] = [x for x in data_dict['trainIm']]
# data_dict['valIm'] = [x for x in data_dict['valIm']]
if args.dataset=='digest':
    num_classes=2
if args.dataset=='iciar':
    num_classes=4
if args.dataset=='colon':
    num_classes=5
input_channel=3
args.top_bn = False
train_dataset = myDataset(  #train_data['trainIm'],
                            #train_data['trainNoiseClass'],
                            filter_result['name'].tolist(),
                            filter_result['correct_label'].tolist(),
                            num_classes, 
                            train=True, 
                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),]),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                        )

test_dataset = myDataset(train_data['valIm'],
                            train_data['valClass'], 
                            num_classes, 
                            train=False, 
                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),]),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                        )
    
if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        output = F.softmax(logit, dim=1)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, event_record_1,event_record_2):
    print ('Training %s...' % model_str)
    print ('############co-teaching#########')
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
   
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        logits1=model1(images).logits
        # record_history(ind,logits1,labels,event_record_1)
        prec1,_ = accuracy(logits1, labels, topk=(1,1))
        train_total+=1
        train_correct+=prec1
        logits2 = model2(images).logits
        # record_history(ind,logits2,labels,event_record_2)
        prec2,_ = accuracy(logits2, labels, topk=(1,1))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        # loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coweight(logits1, logits2, labels, epoch, rate_schedule[epoch], ind, noise_or_not, event_record_1,event_record_2)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward(retain_graph=True)#retain_graph=True
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1[0].cpu().numpy().sum(), prec2[0].cpu().numpy().sum(), loss_1.item(), loss_2.item(), 
                  np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print ('Evaluating %s...' % model_str)
    model1.eval()
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            logits1 = model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        model2.eval()
        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            logits2 = model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
    
        acc1 = 100*float(correct1)/float(total1)
        acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    # Define models
    print ('building model...')
    cnn1 = torchvision.models.__dict__['googlenet'](pretrained=args.pretrained,num_classes=2)
    # num_ftrs = cnn1.fc.in_features
    # cnn1.fc = nn.Linear(num_ftrs, num_classes)
    # cnn1 = torchvision.models.resnet34(num_classes=num_classes)
    cnn1.cuda()
    # print (cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    # lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, 10, eta_min=0, last_epoch=-1)
    # cnn2 = torchvision.models.resnet34(num_classes=num_classes)
    cnn2 = torchvision.models.__dict__['googlenet'](pretrained=args.pretrained,num_classes=2)
    # num_ftrs = cnn2.fc.in_features
    # cnn2.fc = nn.Linear(num_ftrs, num_classes)
    cnn2.cuda()
    # print (cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)
    # lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, 10, eta_min=0, last_epoch=-1)
    mean_pure_ratio1=0
    mean_pure_ratio2=0

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    # test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    # print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # # save results
    # with open(txtfile, "a") as myfile:
    #     myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        # adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        # adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2,event_record_1,event_record_2)
        # lr_scheduler1.step()
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # test_acc12, test_acc22 = evaluate(test_loader2, cnn1, cnn2)
        # lr_scheduler2.step()
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))

if __name__=='__main__':
    main()
