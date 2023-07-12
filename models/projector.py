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


class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        x = self.ext(x)
        return self.head(x)

class Projector(nn.Module):
    def __init__(self, extractor, channels, size, K=10):
        super(Projector, self).__init__()
        self.extractor = extractor
        self.projector = nn.Linear(channels*size*size, K)

        for k,v in enumerate(self.extractor.parameters()):
            v.requires_grad_(False)

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        z = self.projector(x)
        return z
        #return F.softmax(z)
