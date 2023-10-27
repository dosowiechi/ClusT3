import torch
import torch.nn as nn
import torch.nn.functional as F


def get_part(model,layer):
    if layer ==1:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1]
    elif layer ==2:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2]
    elif layer ==3:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3]
    elif layer ==4:
        extractor = [model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3, model.layer4]
    return nn.Sequential(*extractor)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, K=10, multi=5, projector=[None, None, None, None]):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) #64
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) #128
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) #256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) #512
        self.fc = nn.Linear(512*block.expansion, num_classes) #512
        if projector[0] is not None:
            self.projector1 = nn.ModuleList()
            for k in range(multi):
                self.projector1.append(nn.Conv2d(256, K, kernel_size=1)) #Linear(256*32*32, num_classes)
        if projector[1] is not None:
            self.projector2 = nn.ModuleList()
            if multi==1:
                self.projector2.append(nn.Sequential(nn.Conv2d(512, int(512/2), kernel_size=1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(int(512/2), K, kernel_size=1)))
            else:
                for k in range(multi):
                     self.projector2.append(nn.Conv2d(512, K, kernel_size=1)) #Linear(512*16*16, num_classes)
        if projector[2] is not None:
            self.projector3 = nn.ModuleList()
            for k in range(multi):
                self.projector3.append(nn.Conv2d(1024, K, kernel_size=1)) #Linear(1024*8*8, num_classes)
        if projector[3] is not None:
            self.projector4 = nn.ModuleList()
            for k in range(multi):
                self.projector4.append(nn.Conv2d(2048, K, kernel_size=1)) #Linear(2048*4*4, num_classes)

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False, projection=False):
        out_proj = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if hasattr(self, 'projector1') and projection:
            for proj in self.projector1:
                z1 = proj(out)
                z1 = z1.view(z1.size(1), -1)
                out_proj.append(z1)
        out = self.layer2(out)
        if hasattr(self, 'projector2') and projection:
            for proj in self.projector2:
                z2 = proj(out)
                z2 = z2.view(z2.size(1), -1)
                out_proj.append(z2)
        out = self.layer3(out)
        if hasattr(self, 'projector3') and projection:
            for proj in self.projector3:
                z3 = proj(out)
                z3 = z3.view(z3.size(1), -1)
                out_proj.append(z3)
        out = self.layer4(out)
        if hasattr(self, 'projector4') and projection:
            for proj in self.projector4:
                z4 = proj(out)
                z4 = z4.view(z4.size(1), -1)
                out_proj.append(z4)
        # out = F.avg_pool2d(out, 4)
        out = self.AdaptiveAvgPool(out)
        features = out
        out = self.fc(out.squeeze())
        if feature:
            if projection:
                return out, features, out_proj
            else:
                return out, features
        else:
            if projection:
                return out, out_proj
            else:
                return out

def resnet50(num_classes = 10, projector = [None, None, None, None], K=10, multi=5,  **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, K=K, multi=multi, projector=projector, **kwargs)
    return model
