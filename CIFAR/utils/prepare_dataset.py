import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

augment_transforms = transforms.Compose([transforms.RandomRotation(180),
                                         transforms.ColorJitter()])


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_test_data(args):
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
            teset.data = teset_raw

        elif args.corruption == 'cifar_new':
            from CIFAR.utils.cifar_new import CIFAR_New
            teset = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/', transform=te_transforms)
            permute = False
        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=False, transform=te_transforms)

            teset.data = teset_raw
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        te_sampler = torch.utils.data.distributed.DistributedSampler(teset)
    else:
        te_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    if args.distributed:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
            shuffle=(te_sampler is None), num_workers=args.workers, pin_memory=True, sampler=te_sampler)
    else:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)

    return teloader, te_sampler


def prepare_train_data(args):
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=True, download=False, transform=tr_transforms)
    elif args.dataset == 'cifar100':
        trset = torchvision.datasets.CIFAR100(root=args.dataroot, train=True, download=False, transform=tr_transforms)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(trset)
    else:
        tr_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
        shuffle=(tr_sampler is None), num_workers=args.workers, pin_memory=True, sampler=tr_sampler)
    return trloader, tr_sampler
