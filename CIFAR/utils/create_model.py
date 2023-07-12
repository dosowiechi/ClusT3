from CIFAR.models import ResNet_MultiProj


def model_sizes(args, layer):
    if args.dataset == 'imagenet':
        if layer == 0:
            channels, resolution = 64, 112
        if layer == 1:
            channels, resolution = 256, 56
        if layer == 2:
            channels, resolution = 512, 28
        if layer == 3:
            channels, resolution = 1024, 14
        if layer == 4:
            channels, resolution = 2048, 7

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if layer == 0:
            channels, resolution = 64, 32
        if layer == 1:
            channels, resolution = 256, 32
        if layer == 2:
            channels, resolution = 512, 16
        if layer == 3:
            channels, resolution = 1024, 8
        if layer == 4:
            channels, resolution = 2048, 4

    return channels, resolution

#This is the modified forward_features method from the timm model (special case for CIFAR-10/100)
#Ignore the error, as the function checkpoint_seq is out of context, but is correct inside timm model
def create_model(args, proj_layers, weights=None):
    '''
    :param dataset: dataset to use (CIFAR-10, ImageNet)
    :param layers: where to put mask/adapters. Each element with the form (TYPE, SIZE)
    :return: timm model + MaskUp
    '''
    #Creating model based on dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        # if args.multi == 1 :
        #     model = ResNet_projector.resnet50(num_classes, proj_layers, K=args.K)
        # if args.multi != 1 :
        model = ResNet_MultiProj.resnet50(num_classes, proj_layers, K=args.K, multi=args.multi)
        #model = ResNet.resnet50(num_classes)
    if args.dataset == 'cifar100':
        num_classes = 100
        # if args.multi == 1 :
        #     model = ResNet_projector.resnet50(num_classes, proj_layers, K=args.K)
        # if args.multi != 1 :
        model = ResNet_MultiProj.resnet50(num_classes, proj_layers, K=args.K, multi=args.multi)
    #Loading weights
    if weights is not None:
        model.load_state_dict(weights, strict=False)

    return model

