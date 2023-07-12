import timm
import types
import torch.nn as nn

def visda_forward_features(self, x, project=False):
    out_proj = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    #if self.grad_checkpointing and not torch.jit.is_scripting():
    #    x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    x = self.layer1(x)
    if hasattr(self, 'projector1'):
        for proj in self.projector1:
            z1 = proj(x)
            z1 = z1.view(z1.size(1), -1)
            out_proj.append(z1)
    x = self.layer2(x)
    if hasattr(self, 'projector2'):
        for proj in self.projector2:
            z2 = proj(x)
            z2 = z2.view(z2.size(1), -1)
            out_proj.append(z2)
    x = self.layer3(x)
    if hasattr(self, 'projector3'):
        for proj in self.projector3:
            z3 = proj(x)
            z3 = z3.view(z3.size(1), -1)
            out_proj.append(z3)
    x = self.layer4(x)
    if hasattr(self, 'projector4'):
        for proj in self.projector4:
            z4 = proj(x)
            z4 = z4.view(z4.size(1), -1)
            out_proj.append(z4)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    if project:
        return x, out_proj
    else:
        return x

def visda_forward(self, x, project=False):
    out_proj = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    #if self.grad_checkpointing and not torch.jit.is_scripting():
    #    x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    x = self.layer1(x)
    if hasattr(self, 'projector1'):
        z1 = self.projector1(x)
        z1 = z1.view(z1.size(1), -1)
        out_proj.append(z1)
    x = self.layer2(x)
    if hasattr(self, 'projector2'):
        z2 = self.projector2(x)
        z2 = z2.view(z2.size(1), -1)
        out_proj.append(z2)
    x = self.layer3(x)
    if hasattr(self, 'projector3'):
        z3 = self.projector3(x)
        z3 = z3.view(z3.size(1), -1)
        out_proj.append(z3)
    x = self.layer4(x)
    if hasattr(self, 'projector4'):
        z4 = self.projector4(x)
        z4 = z4.view(z4.size(1), -1)
        out_proj.append(z4)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    if project:
        return x, out_proj
    else:
        return x

def model_sizes(dataset, layer):
    if dataset == 'imagenet':
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

    elif dataset == 'cifar10' or dataset == 'cifar100':
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

    elif dataset == 'visda' or dataset == 'office':
        if layer == 0:
            channels, resolution = 64, 32
        if layer == 1:
            channels, resolution = 256, 112
        if layer == 2:
            channels, resolution = 512, 56
        if layer == 3:
            channels, resolution = 1024, 28
        if layer == 4:
            channels, resolution = 2048, 14

    return channels, resolution

class Projector(nn.Module):
    def __init__(self, channels, K, size='small'):
        super(Projector, self).__init__()
        if size == 'small':
            self.proj = nn.Conv2d(channels, K, kernel_size=1)
        else:
            self.proj = nn.Sequential(nn.Conv2d(channels, int(channels/2), kernel_size=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(int(channels/2), K, kernel_size=1))
    def forward(self, x):
        return self.proj(x)

def create_model(args, projectors=(False,False,False,False), weights=None):
    '''
    :param args: configuration set
    :param projectors: where to put projector layers in the encoder
    :param weights: pretrained weights
    :return: model (with projectors)
    '''
    #Creating model based on dataset
    if args.dataset == 'visda' or args.dataset == 'office':
        func_type = types.MethodType
        if args.dataset == 'visda':
            num_classes = 12
        else:
            num_classes = 65
        model = timm.create_model('resnet50', num_classes=num_classes, features_only=True, pretrained=False)
        model.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(2048, num_classes)
        if args.multi > 1:
            model.forward = func_type(visda_forward_features, model)
        else:
            model.forward = func_type(visda_forward, model)
        if projectors[0]:
            channels, _ = model_sizes(args.dataset, layer=1)
            if args.multi > 1:
                model.projector1 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector1.append(Projector(channels, args.K, args.psize))
            else:
                model.projector1 = Projector(channels, args.K, args.psize)
        if projectors[1]:
            channels, _ = model_sizes(args.dataset, layer=2)
            if args.multi > 1:
                model.projector2 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector2.append(Projector(channels, args.K, args.psize))
            else:
                model.projector2 = Projector(channels, args.K, args.psize)
        if projectors[2]:
            channels, _ = model_sizes(args.dataset, layer=3)
            if args.multi > 1:
                model.projector3 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector3.append(Projector(channels, args.K, args.psize))
            else:
                model.projector3 = Projector(channels, args.K, args.psize)
        if projectors[3]:
            channels, _ = model_sizes(args.dataset, layer=4)
            if args.multi > 1:
                model.projector4 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector4.append(Projector(channels, args.K, args.psize))
            else:
                model.projector4 = Projector(channels, args.K, args.psize)
    else:
        print('Dataset not found! Model could not be created')

    #Loading weights
    if weights is not None:
        if args.dataset == 'visda' or args.dataset == 'office':
            del weights['fc.weight']
            del weights['fc.bias']
        model.load_state_dict(weights, strict=False)

    return model

def create_model2(args, projectors=(False,False,False,False), weights=None):
    '''
    :param args: configuration set
    :param projectors: where to put projector layers in the encoder
    :param weights: pretrained weights
    :return: model (with projectors)
    '''
    #Creating model based on dataset
    if args.dataset == 'visda' or args.dataset == 'office':
        func_type = types.MethodType
        if args.dataset == 'visda':
            num_classes = 12
        else:
            num_classes = 65
        model = timm.create_model('resnet50', num_classes=num_classes, features_only=True, pretrained=False)
        model.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(2048, num_classes)
        if args.multi > 1:
            model.forward = func_type(visda_forward_features, model)
        else:
            model.forward = func_type(visda_forward, model)
        if projectors[0]:
            channels, _ = model_sizes(args.dataset, layer=1)
            if args.multi > 1:
                model.projector1 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector1.append(nn.Conv2d(channels, num_classes, kernel_size=1))
            else:
                model.projector1 = nn.Conv2d(channels, num_classes, kernel_size=1)
        if projectors[1]:
            channels, _ = model_sizes(args.dataset, layer=2)
            if args.multi > 1:
                model.projector2 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector2.append(nn.Conv2d(channels, num_classes, kernel_size=1))
            else:
                model.projector2 = nn.Conv2d(channels, num_classes, kernel_size=1)
        if projectors[2]:
            channels, _ = model_sizes(args.dataset, layer=3)
            if args.multi > 1:
                model.projector3 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector3.append(nn.Conv2d(channels, num_classes, kernel_size=1))
            else:
                model.projector3 = nn.Conv2d(channels, num_classes, kernel_size=1)
        if projectors[3]:
            channels, _ = model_sizes(args.dataset, layer=4)
            if args.multi > 1:
                model.projector4 = nn.ModuleList()
                for k in range(args.multi):
                    model.projector4.append(nn.Conv2d(channels, num_classes, kernel_size=1))
            else:
                model.projector4 = nn.Conv2d(channels, num_classes, kernel_size=1)
    else:
        print('Dataset not found! Model could not be created')

    #Loading weights
    if weights is not None:
        if args.dataset == 'visda' or args.dataset == 'office':
            del weights['fc.weight']
            del weights['fc.bias']
        model.load_state_dict(weights, strict=False)

    return model


