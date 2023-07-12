import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils import create_model, prepare_dataset, utils
import configuration

best_acc1 = 0

def main(args):
    global best_acc1
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)
    if rank == 0:
        print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    if rank == 0:
        print('From Rank: {}, ==> Making model..'.format(rank))

    weights = torch.load(args.root + '/weights/resnet50_imagenet.pth')
    model = create_model.create_model(args, weights=weights).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(rank))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if rank == 0:
            print("=> loaded checkpoint '{}' (epoch {})".format(rank, checkpoint['epoch']))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if rank == 0:
        print('From Rank: {}, ==> Preparing data..'.format(rank))
    cudnn.benchmark = True
    trloader, trsampler, teloader, tesampler = prepare_dataset.prepare_train_data(args)
    if rank == 0:
        print('Test on original data')

    if rank == 0:
        print('\t\tTrain Loss \t\t Val Loss \t\t Train Accuracy \t\t Val Acccuracy')

    for epoch in range(args.start_epoch, args.epochs):
        trsampler.set_epoch(epoch)
        tesampler.set_epoch(epoch)
        acc_train, loss_train = train(model, criterion, optimizer, trloader, args.ignore_batch)
        acc_val, loss_val = validate(model, criterion, teloader)

        if rank == 0:
            print(('Epoch %d/%d:' % (epoch, args.epochs)).ljust(24) +
                      '%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train, loss_val, acc_train, acc_val))

        best_acc1 = max(acc_val, best_acc1)

        if rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                }, args)

def train(model, criterion, optimizer, train_loader, ignore_stats):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    if ignore_stats:
        model.eval()
    else:
        model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        #Compute output and loss
        output = model(images)
        loss = criterion(output, target)

        #Compute accuracy
        acc1 = utils.accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)

    return top1.avg, losses.avg


def validate(model, criterion, val_loader):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = utils.accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, losses.avg


if __name__=='__main__':
    args = configuration.argparser()
    main(args)