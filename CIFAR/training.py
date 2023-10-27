import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import math

from utils import create_model, prepare_dataset, utils
import configuration


best_uns = math.inf
lbd = 1

def main(args):
    global best_uns
    global lbd
    # checkpoint = torch.load(args.root + 'weights/BestClassification.pth')
    # weights = checkpoint['net']

    args.corruption = 'original'


    proj_layer = [args.project_layer1, args.project_layer2, args.project_layer3, args.project_layer4]
    model = create_model.create_model(args, proj_layer).cuda()
    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.dataset == 'tiny-imagenet':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(150, 250), gamma=0.1)

    # model.train()
    # model.module.projector.train()

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True
    teloader, tesampler = prepare_dataset.prepare_test_data(args)
    trloader, trsampler = prepare_dataset.prepare_train_data(args)
    print('Test on original data')

    print('\t\tTrain Loss \t\t Train Accuracy \t\t Val Loss \t\t Val Accuracy')

    for epoch in range(args.start_epoch, args.epochs):
        acc_train, loss_train = train(model, optimizer, trloader, args)
        acc_val, loss_val = validate(model, teloader, args)
        scheduler.step()

        print(('Epoch %d/%d:' % (epoch, args.epochs)).ljust(24) +
                      '%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train, acc_train, loss_val, acc_val))

        is_best = loss_val < best_uns
        best_uns = max(loss_val, best_uns)


        if proj_layer[0] != None:
                dict = {
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'projector': model.projector1.state_dict(),
                    'best_uns': best_uns,
                    'optimizer': optimizer.state_dict(),
                }
        if proj_layer[1] != None:
                dict = {
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'projector': model.projector2.state_dict(),
                    'best_uns': best_uns,
                    'optimizer': optimizer.state_dict(),
                }
        if proj_layer[2] != None:
                dict = {
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'projector': model.projector3.state_dict(),
                    'best_uns': best_uns,
                    'optimizer': optimizer.state_dict(),
                }
        if proj_layer[3] != None:
                dict = {
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'projector': model.projector4.state_dict(),
                    'best_uns': best_uns,
                    'optimizer': optimizer.state_dict(),
                }
        utils.save_checkpoint(dict, is_best, args)

def train(model, optimizer, train_loader, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    lbd = args.lbd
    model.train()
    end = time.time()
    entropy = utils.Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    cross_entropy = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        #Compute output and loss
        output, out_proj = model(images, projection = True)
        loss = 0.0
        for i, layer in enumerate(out_proj):
            if layer != None:
                x_mean = layer.mean(dim=1)
                loss += lbd*(entropy(layer) + kl(nn.functional.log_softmax(x_mean, dim=0), torch.full_like(x_mean, 1 / args.K)))
        loss += cross_entropy(output, target)

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


def validate(model, val_loader, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    lbd = args.lbd
    model.eval()
    entropy = utils.Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    cross_entropy = nn.CrossEntropyLoss()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, out_proj = model(images, projection=True)
            loss = 0.0
            for i, layer in enumerate(out_proj):
                if layer != None:
                    x_mean = layer.mean(dim=1)
                    loss += lbd*(entropy(layer) + kl(nn.functional.log_softmax(x_mean, dim=0),
                                               torch.full_like(x_mean, 1 / args.K)))
            loss += cross_entropy(output, target)

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
