import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.backends import cudnn
from torch.utils.data import distributed, DataLoader
from torchvision import transforms, datasets

from nets import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.cfg['policy'] == 'step':
        if isinstance(args.cfg['step'], int):
            msg = 'int'
            lr = args.cfg['lr'] * (args.cfg['gamma'] ** (epoch // args.cfg['step']))
        elif isinstance(args.cfg['step'], list):
            msg = 'list'
            r = 0
            for step in args.cfg['step']:
                if epoch + 1 > step:
                    r += 1
            lr = args.cfg['lr'] * args.cfg['gamma'] ** r
        else:
            msg = 'default, int=30'
            lr = args.cfg['lr'] * (0.1 ** (epoch // 30))
    elif args.cfg['policy'] == 'cos':
        msg = 'cosine'
        lr = 0.5 * args.cfg['lr'] * (1 + math.cos(math.pi * epoch / args.cfg['max_epochs']))
    else:
        msg = 'fixed'
        lr = args.cfg['lr']

    if args.local_rank in {-1, 0}:
        print('lr = %f, %s' % (lr, msg))

    optimizer.param_groups[0]['lr'] = lr


def save_checkpoint(state, is_best, directory, filename='/last_checkpoint.pth.tar'):
    filename = directory + filename
    if os.path.exists(filename):
        os.remove(filename)
    torch.save(state, filename)
    if is_best:
        if os.path.exists(directory + '/model_best.pth'):
            os.remove(directory + '/model_best.pth')
        torch.save(state, directory + '/model_best.pth.tar')


def data_save(root, epoch, file):
    with open(root, "a") as f:
        f.write(str(epoch) + " " + str(file) + '\n')


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(args.device, non_blocking=True)
            targets = targets.cuda(args.device, non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.local_rank in {-1, 0}:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        if args.local_rank in {-1, 0}:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.device, non_blocking=True)
        targets = targets.cuda(args.device, non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, targets)
        assert torch.isfinite(loss).any()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank in {-1, 0}:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def increment_path(path, exist_ok=False, sep='-'):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    return path


def parameter():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-c', '--cfg', type=str, metavar='PATH', help='path to schedules', required=True)
    parser.add_argument('-p', '--path', type=str, metavar='PATH', help='location of the data corpus', required=True)
    parser.add_argument('-s', '--set', type=str, help='name of the dataset', default='imagenet')
    parser.add_argument('-a', '--att', type=str, help='name of the attention module', default=None)
    parser.add_argument('-n', '--net', '-n', type=str, help='name of the network architecture', default='ResNet50')
    parser.add_argument('--seed', type=int, help='seed for initializing training. ')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--device', default=None, type=int, help='GPU/NPU id to use.')
    parser.add_argument('--print-freq', '-f', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--action', default='exp', type=str, help='other information')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int, help='do not modify')
    args = parser.parse_args()

    with open(args.cfg, errors='ignore') as f:
        args.cfg = yaml.safe_load(f)  # load cfg dict

    if args.local_rank in {-1, 0}:
        if args.resume:
            args.directory = args.resume.rsplit('/', 1)[0]
        else:
            args.directory = "outputs/%s/%s/%s/%s/" % (args.set, args.net, str(args.att), args.action)
            args.directory = str(increment_path(args.directory, exist_ok=False))

        if not os.path.exists(args.directory):
            os.makedirs(args.directory)

        # shutil.copyfile('modules/tfam.py', directory + '/tfam.py')

        print('saved dir: ', args.directory)
        print(args, args.cfg)

    return args


def main():
    best_epoch = 0
    best_prec1 = 0
    best_prec5 = 0

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        cudnn.benchmark = False
        cudnn.deterministic = True

        if args.local_rank in {-1, 0}:
            print('Set random seed to {}, benchmark: {}, deterministic: {}'.format(args.seed, cudnn.benchmark,
                                                                                   cudnn.deterministic))

    # create model
    if args.net[:6] == "ResNet":
        if args.set == "cifar100":
            model = ResidualNet('cifar100', args.net, 100, args.att)
        elif args.set == "cifar10":
            model = ResidualNet('cifar10', args.net, 10, args.att)
        else:
            model = ResidualNet('imagenet', args.net, 1000, args.att)
    elif args.net[:6] == "resnet":
        if args.set == "cifar100":
            model = resnet_cifar('cifar100', args.net, 100, args.att)
        elif args.set == "cifar10":
            model = resnet_cifar('cifar10', args.net, 10, args.att)
        else:
            model = resnet_cifar('imagenet', args.net, 1000, args.att)
    elif args.net == "MobileNetV2":
        if args.set == "cifar100":
            model = MobileNetV2(args.att, 100)
        elif args.set == "cifar10":
            model = MobileNetV2(args.att, 10)
        else:
            model = MobileNetV2(args.att, 1000)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    args.distributed = False
    if args.local_rank != -1:
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl')
        print('Use DDP mode.')

        model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif args.device is not None:
        print('You have chosen a specific device.')
        model = model.cuda(args.device)
    else:
        model = torch.nn.DataParallel(model).cuda()
        args.device = torch.device("cuda")
        print('Use DP mode.')

    if args.local_rank in {-1, 0}:
        print(model)
        # get the number of models parameters
        print('Number of models parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.cfg['lr'],
                                momentum=args.cfg['momentum'],
                                weight_decay=args.cfg['weight_decay'])

    # optionally resume from a checkpoint
    if args.resume and args.local_rank in {-1, 0}:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    args.data = args.path + '/' + args.set
    if args.set == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.set == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageNet(root=args.data, split="train", transform=train_transform)
        valid_dataset = datasets.ImageNet(root=args.data, split="val", transform=valid_transform)

    if args.distributed:
        train_sampler = distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.cfg['batch_size'], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.cfg['batch_size'], shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        m = time.time()
        _, _ = validate(val_loader, model, criterion)
        n = time.time()
        if args.local_rank in {-1, 0}:
            print((n - m) / 3600)
        return

    for epoch in range(args.start_epoch, args.cfg['max_epochs']):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        loss_temp, train_prec1_temp, train_prec5_temp = train(train_loader, model, criterion, optimizer, epoch)

        prec1, prec5 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank in {-1, 0}:
            is_best = prec1 > best_prec1
            if is_best:
                best_epoch = epoch
                best_prec1 = prec1
                best_prec5 = prec5
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.directory, args.cfg['max_epochs'])

            data_save(args.directory + '/Loss_plot.txt', epoch, loss_temp)
            data_save(args.directory + '/train_prec1.txt', epoch, train_prec1_temp)
            data_save(args.directory + '/train_prec5.txt', epoch, train_prec5_temp)
            data_save(args.directory + '/val_prec1.txt', epoch, prec1)
            data_save(args.directory + '/val_prec5.txt', epoch, prec5)

            end_time = time.time()
            time_value = (end_time - start_time) / 60

            print("-" * 80)
            print(time_value)
            print("-" * 80)

        # scheduler.step()

    if args.local_rank in {-1, 0}:
        print('Best Result: Prec@1 {} Prec@5 {} eopch {}'.format(best_prec1, best_prec5, best_epoch))
        print('Save Dir: ', args.directory)


if __name__ == '__main__':
    args = parameter()
    bt = time.time()
    main()
    et = time.time()
    if args.local_rank in {-1, 0}:
        print('run times: ', (et - bt) / 3600, 'h')