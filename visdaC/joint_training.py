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
        print('No. of Projectors: ', args.multi)
        print('Projector size: ', args.psize)
        print('No. of Clusters: ', args.K)

    projectors = tuple(map(bool, args.layers))
    weights = torch.load(args.root + '/weights/resnet50_imagenet.pth')
    model = create_model.create_model(args, projectors=projectors, weights=weights).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    classes = 12

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if rank == 0:
        print('Using Optimizer: ', args.optim)

    if args.resume:
        ckpt = args.dataset +'_' + utils.list_to_str(args.layers) + '_W' + str(args.gamma) + '_N' + str(args.multi) + '_S' + args.psize + '_K' + str(args.K) + '.pth'
        checkpoint = torch.load(args.root + '/weights/' + ckpt)
        if args.distributed:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']

    if rank == 0:
        print('From Rank: {}, ==> Preparing data..'.format(rank))
    cudnn.benchmark = True
    trloader, trsampler, teloader, tesampler = prepare_dataset.prepare_train_data(args)
    if rank == 0:
        print('Test on original data')

    if rank == 0:
        print('\t\tTrain Loss \t\t CE Loss \t\t IM Loss \t\t Val Loss \t\t Train Accuracy \t\t Val Acccuracy')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        trsampler.set_epoch(epoch)
        tesampler.set_epoch(epoch)
        acc_train, loss_train, loss_c, loss_p = train(model, classes, criterion, optimizer, trloader)
        acc_val, loss_val = validate(model, classes, criterion, teloader)

        if rank == 0:
            print(('Epoch %d/%d:' % (epoch, args.epochs)).ljust(24) +
                      '%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train, loss_c, loss_p, loss_val, acc_train, acc_val))

        best_acc1 = max(acc_val, best_acc1)

        if rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                }, args)
    end_time = time.time()
    training_time = end_time - start_time
    if rank == 0:
        print('Total Training time: ', training_time)

def train(model, num_classes, criterion, optimizer, train_loader):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_c = utils.AverageMeter('Crossentropy', ':.4e')
    losses_p = utils.AverageMeter('IM', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()
    entropy = utils.Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        #Compute output and loss
        output, projections = model(images, project=True)
        loss_p = 0.0
        for proj in projections:
            mean_proj = proj.mean(dim=0)
            loss_p += entropy(proj) + kl(mean_proj.log_softmax(dim=0), torch.full_like(mean_proj, 1 / num_classes))
        loss_c = criterion(output, target)
        loss = loss_c + args.gamma*loss_p

        #Compute accuracy
        acc1 = utils.accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        losses_c.update(loss_c.item(), images.size(0))
        losses_p.update(loss_p.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)

    return top1.avg, losses.avg, losses_c.avg, losses_p.avg


def validate(model, num_classes, criterion, val_loader):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        entropy = utils.Entropy()
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, projections = model(images, project=True)
            loss_p = 0.0
            for proj in projections:
                mean_proj = proj.mean(dim=0)
                loss_p += entropy(proj) + kl(mean_proj.log_softmax(dim=0), torch.full_like(mean_proj, 1 / num_classes))
            loss = criterion(output, target) + args.gamma*loss_p

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