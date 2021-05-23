from __future__ import print_function
import datetime
import os
import time
import sys
import pickle
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.utils import shuffle
import loadData as ld
import DataSet as myDataLoader
import utils
import numpy as np
import json
import torch.nn.functional as F
import pandas as pd
try:
    from apex import amp
except ImportError:
    amp = None

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def record_history(index,output,target,recorder):
    pred = F.softmax(output, dim=1).cpu().data
    # pred = output.cpu().data
    # _, pred = torch.max(F.softmax(output, dim=1).data, 1)
    for i,ind in enumerate(index):
        recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())
        ## save forget event 
        # recorder[ind].append((target.cpu()[i] == pred.cpu()[i]).numpy().tolist())
    return

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    input = input.unsqueeze(1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)
    return result

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, recorder, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target, ind in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image).logits
        record_history(ind,output,target,recorder)
        # m = nn.Sigmoid()
        loss = criterion(output, target)
        ## weighted loss
        # m_prob = np.array([np.mean(recorder[ind[i]]) for i in range(len(ind))])
        # prob_1 = torch.gather(F.softmax(output, dim=1),1,target.view(target.size()[0],1))
        # prob_1 = prob_1.view(target.size()[0])
        # weight = torch.pow(1-prob_1,2)/torch.sum(torch.pow(1-prob_1,2))
        # loss = F.cross_entropy(output, target, reduce = False)*weight
        # loss = torch.sum(loss)
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        acc1 = utils.accuracy(output, target, topk=(1, ))[0]
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    correct = 0
    total = 0
    header = 'Test:'
    with torch.no_grad():
        for image, target,_ in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            _, pred1 = torch.max(F.softmax(output, dim=1).data, 1)
            m = nn.Sigmoid()
            loss = criterion(output,target)
            # loss = criterion(m(output), make_one_hot(target,2))
            acc1 = utils.accuracy(output, target, topk=(1, ))[0]
            batch_size = image.shape[0]
            total+=batch_size
            correct += (pred1.cpu()==target.cpu()).sum()
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('accuracy :{}'.format(100*float(correct)/float(total)))
    print(' * Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg

def main(args):
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    filter_result = pd.read_csv('iteration/bishe/corrected_digest_40.csv')
    # filter_result=shuffle(filter_result,random_state=20)
    # train_data = pickle.load(open('pickle_data/digest_40.p',"rb"))
    val_data = pickle.load(open('pickle_data/digest_40.p', "rb"))
    # n_train = len(train_data['trainClass'])
    n_train = len(filter_result)
    print('training examples:', n_train)
    event_record = [[] for i in range(n_train)]
    trainDataset = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])

    valDataset = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    # dataset = myDataLoader.MyDataset(train_data['trainIm'],train_data['trainClass'],transform=trainDataset)
    dataset = myDataLoader.MyDataset(filter_result['name'].tolist(),filter_result['correct_label'].tolist(),transform=trainDataset)
    dataset_test = myDataLoader.MyDataset(val_data['valIm'],val_data['valClass'],transform=valDataset)
    print(len(filter_result[filter_result['y']==filter_result['correct_label']]),len(filter_result))
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, drop_last=True,shuffle=False, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,drop_last=True,shuffle=False, pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.classes)  
    model.to('cuda')
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    ## adjust lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, event_record, args.apex)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
    ## save training history
    # INCV_results = pd.DataFrame({'name':train_data['trainIm'],
    #                         #  'y_noisy':[int(x) for x in data['trainNoiseClass']],
    #                         # 'y_noisy':filter_result['correct_label'].tolist(),
    #                          'y':train_data['trainClass'],
    #                          'logits':event_record})

    # INCV_results.to_csv(
    #         os.path.join(args.output_dir,'colon.csv'),
    #         index = False,
    #         columns = ['name','y','logits'])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data_dir', default="/home/pengting/Documents/miccai/miccai", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=224, help='Width of the input patch')
    parser.add_argument('--inHeight', type=int, default=224, help='Height of the input patch')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--model', default='googlenet', help='model')#densenet161,resnet101,resnext101_32x8d
    parser.add_argument('--resume_model', default='', help='pretrained model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch-size', default=180, type=int)
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output_dir', default='iteration/bishe', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument('--cached_data_file', default='digest_0.4.p', help='Data file names and other values, such as'
                                                                           'class weights, are cached')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
