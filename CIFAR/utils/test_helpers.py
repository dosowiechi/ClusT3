import numpy as np
import torch
import torch.nn as nn

from models import ResNet_projector

def build_model(args, state_dict=None):
	print('Building model...')
	args.projection = False
	num_classes = 10
	net = ResNet_projector.resnet50(num_classes).cuda()
	if state_dict is not None:
		print('Using pre-trained classifier')
		net.load_state_dict(state_dict)
	if args.project_layer1 != None:
		ext = get_part(net, 1)
		head = nn.Conv2d(256, num_classes, kernel_size=1)
	if args.project_layer2 != None:
		ext = get_part(net, 2)
		head = nn.Conv2d(512, num_classes, kernel_size=1)
	if args.project_layer3 != None:
		ext = get_part(net, 3)
		head = nn.Conv2d(1024, num_classes, kernel_size=1)
	if args.project_layer4 != None:
		ext = get_part(net, 4)
		head = nn.Conv2d(2048, num_classes, kernel_size=1)
	uns = ExtractorHead(ext, head).cuda()

	return net, ext, head, uns


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