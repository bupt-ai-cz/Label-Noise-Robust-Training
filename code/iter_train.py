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
import loadData as ld
import DataSet as myDataLoader
from sklearn.utils import shuffle
import utils
import numpy as np
import json
import torch.nn.functional as F
import pandas as pd
from bootstrap_loss import HardBootstrappingLoss
try:
    from apex import amp
except ImportError:
    amp = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def record_history(index,output,target,pro_recorder):
    pred = F.softmax(output, dim=1).cpu().data
    for i,ind in enumerate(index):
        pro_recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, pro_recorder, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target, ind in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image).logits
        # m = nn.Sigmoid()
        # if epoch > 2:
        #     loss = criterion(output, target, ind, pro_recorder)
        # else:
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # record_history(ind,output,target,pro_recorder)
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
            acc1 = utils.accuracy(output, target, topk=(1, ))[0]
            batch_size = image.shape[0]
            total+=batch_size
            correct += (pred1.cpu()==target.cpu()).sum()
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
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
    # data = pickle.load(open('pickle_data/digest_20.p', "rb"))
    def meanStr(line):
        num_list = json.loads(line['logits'])[:30]#logits
        num_list = np.log10(num_list)
        return np.mean(num_list)
    ## detect label noise and drop it
    # filter_result = pd.read_csv(os.path.join(args.output_dir,'digest_0.2_iter1.csv'))
    # filter_result['mean'] = filter_result.apply(meanStr,axis=1)
    # thred_40 = filter_result.sort_values(by=['mean'])['mean'].iloc[int(len(filter_result)*0.4)]
    # thred_0 = filter_result.loc[filter_result['y_noisy']==0].sort_values(by=['mean'])['mean'].iloc[int(len(filter_result.loc[filter_result['y_noisy']==0])*0.4)]
    # thred_1 = filter_result.loc[filter_result['y_noisy']==1].sort_values(by=['mean'])['mean'].iloc[int(len(filter_result.loc[filter_result['y_noisy']==1])*0.4)]
    # thred_2 = filter_result.loc[filter_result['y_noisy']==2].sort_values(by=['mean'])['mean'].iloc[int(len(filter_result.loc[filter_result['y_noisy']==2])*0.51)]
    # thred_3 = filter_result.loc[filter_result['y_noisy']==3].sort_values(by=['mean'])['mean'].iloc[int(len(filter_result.loc[filter_result['y_noisy']==3])*0.51)]
    # filter_result_0 = filter_result.loc[(filter_result['y_noisy']==0)&(filter_result['mean']>thred_0)][['name','y','y_noisy']].copy()
    # filter_result_1 = filter_result.loc[(filter_result['y_noisy']==1)&(filter_result['mean']>thred_1)][['name','y','y_noisy']].copy()
    # filter_result_2 = filter_result.loc[(filter_result['y_noisy']==2)&(filter_result['mean']>thred_2)][['name','y','y_noisy']].copy()
    # filter_result_3 = filter_result.loc[(filter_result['y_noisy']==3)&(filter_result['mean']>thred_3)][['name','y','y_noisy']].copy()
    # filter_result = pd.concat([filter_result_0,filter_result_1],sort=False)
    # filter_result = filter_result[filter_result['mean']>thred_40]
    # print(len(filter_result))
    # filter_result['select'] = filter_result.apply(lambda x:False if x.mean>thred_20 else True,axis=1)
    # filter_result = filter_result[filter_result['select']==True]
    # print(np.sum(filter_result['select'].values),len(filter_result))
    # filter_result.loc[(filter_result['meanEvents']>0.487)&(filter_result['pre_label']==1),'select']=False
    # filter_result = filter_result[filter_result['select']==True]
    # train_idx = filter_result['select'].values
    # filter_result=shuffle(filter_result)
    # y_train = filter_result['y'].values
    # y_train_noise = filter_result['y_noisy'].values
    # noise_or_not = (y_train==y_train_noise)[train_idx]
    if not os.path.isfile(args.cached_data_file):
        dataLoader = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        if dataLoader is None:
            print('Error while processing the data. Please check')
            exit(-1)
        data = dataLoader.processData()
    else:
        print('load cached file')
        data = pickle.load(open(args.cached_data_file, "rb"))
    n_train = len(data['trainIm'])
    prob_record = [[] for i in range(n_train)]
    trainDataset = transforms.Compose([
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # normalize
        ])

    valDataset = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # normalize,
    ])
    # total_sample = np.sum(filter_result['name'][train_idx])
    # print(np.sum(filter_result['label'].values),n_train)
    dataset = myDataLoader.MyDataset(data['trainIm'],data['trainNoiseClass'],transform=trainDataset)
    dataset_test = myDataLoader.MyDataset(data['valIm'],data['valClass'],transform=valDataset)
    # dataset_test = myDataLoader.MyDataset(filter_result['name'].tolist(),filter_result['y'].tolist(),transform=valDataset)
    # print(len(filter_result[filter_result['y']==filter_result['correct_label']]),len(filter_result))
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, drop_last=False,shuffle=False, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,drop_last=False,shuffle=False, pin_memory=True)

    print("Creating model")
    # state_dict = torch.load(args.resume_model)
    # loaded_model = state_dict['model']
    # model = torchvision.models.resnext101_32x8d(num_classes=2)
    # model = torchvision.models.resnet101(num_classes=2)
    # model = torchvision.models.resnet34(num_classes=2)
    # model.load_state_dict(loaded_model)    
    # import pdb
    # pdb.set_trace()
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=2)
    #googlenet
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.classes)   
    ##densenet161
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, args.classes)

    ##resnet101
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.classes)  
    # 
    # vgg
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, args.classes)
    model.to('cuda')
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion_before = nn.CrossEntropyLoss()
    # criterion =  HardBootstrappingLoss(beta=0.8)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

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
        train_one_epoch(model, criterion_before, optimizer, data_loader, device, epoch, args.print_freq, prob_record, args.apex)
        lr_scheduler.step()
        evaluate(model, criterion_before, data_loader_test, device=device)
        # if args.output_dir and epoch>55:
        #     checkpoint = {
        #         'model': model_without_ddp.state_dict(),
        #         # 'optimizer': optimizer.state_dict(),
        #         # 'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args}
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'checkpoint.pth'))
    INCV_results = pd.DataFrame({'name':data['trainIm'],
                             'y_noisy':[int(x) for x in data['trainNoiseClass']],
                             'y':data['trainClass'],
                             'prob':prob_record,
                             'event':event_record})

    INCV_results.to_csv(
            os.path.join(args.output_dir,'digest_0.2_iter1.csv'),
            index = False,
            columns = ['name','y_noisy','y','prob','event'])#'y_noisy',
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
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
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
    parser.add_argument('--output_dir', default='result/bishe/colon', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
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
