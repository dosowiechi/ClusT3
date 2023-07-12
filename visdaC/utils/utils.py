import torch
import torch.nn as nn

'''-------------------------Saving weights----------------------------'''
def list_to_str(l):
    st = ''
    for e in l:
        st += str(e)
    return st

def save_checkpoint(state, args):
    root = args.save + args.dataset
    proj_list = list_to_str(args.layers)
    if args.projectors:
        root += '_' + proj_list + '_W' + str(args.gamma) + '_N' + str(args.multi) + '_S' + args.psize + '_K' + str(args.K)
    torch.save(state, root + '.pth')

'''-------------------------Averagemeter-----------------------------'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

'''-------------------Getting Projectors Parameters---------------------'''
def get_parameters(mode, project, model, layer=1, only=False):
    parameters = []
    if mode == 'projectors':
        if project[0]:
            parameters += list(model.projector1.parameters())
        if project[1]:
            parameters += list(model.projector2.parameters())
        if project[2]:
            parameters += list(model.projector3.parameters())
        if project[3]:
            parameters += list(model.projector4.parameters())
    elif mode == 'original':
        if layer == 1:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 2:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 3:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 4:
            parameters = list(extract_params(model, layer, only).parameters())
    return parameters

def get_dist_parameters(mode, project, model, layer=1, only=False):
    parameters = []
    if mode == 'projectors':
        if project[0]:
            parameters += list(model.module.projector1.parameters())
        if project[1]:
            parameters += list(model.module.projector2.parameters())
        if project[2]:
            parameters += list(model.module.projector3.parameters())
        if project[3]:
            parameters += list(model.module.projector4.parameters())
    elif mode == 'original':
        if layer == 1:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 2:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 3:
            parameters = list(extract_params(model, layer, only).parameters())
        elif layer == 4:
            parameters = list(extract_params(model, layer, only).parameters())
    elif mode == 'adapt':
        if project[0] and project[1] and not check_all(project):
            parameters = extract_dist_params(model, layer=2)
        if not project[0] and project[1] and project[2] and not check_all(project):
            parameters = extract_dist_params(model, layer=3)
        if not project[0] and not project[1] and project[2] and project[3] and not check_all(project):
            parameters = extract_dist_params(model, layer=4)
        if check_all(project):
            parameters = extract_dist_params(model, layer=4)
        if project[0] and single_true(project):
            parameters = extract_dist_params(model, layer=1)
        if project[1] and single_true(project):
            parameters = extract_dist_params(model, layer=2)
        if project[2] and single_true(project):
            parameters = extract_dist_params(model, layer=3)
        if project[3] and single_true(project):
            parameters = extract_dist_params(model, layer=4)

    return parameters

def extract_params(model, layer, only=False):
    if layer == 1:
        if only:
            layers = model.layer1
        else:
            layers = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1])
    elif layer == 2:
        if only:
            layers = model.layer2
        else:
            layers = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2])
    elif layer == 3:
        if only:
            layers = model.layer3
        else:
            layers = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3])
    elif layer == 4:
        if only:
            layers = model.layer4
        else:
            layers = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3, model.layer4])

    return layers

def extract_dist_params(model, layer, only=False):
    if layer == 1:
        if only:
            layers = model.module.layer1
        else:
            layers = nn.ModuleList([model.module.conv1, model.module.bn1, nn.ReLU(inplace=True), model.module.layer1])
    elif layer == 2:
        if only:
            layers = model.module.layer2
        else:
            layers = nn.ModuleList([model.module.conv1, model.module.bn1, nn.ReLU(inplace=True), model.module.layer1, model.module.layer2])
    elif layer == 3:
        if only:
            layers = model.module.layer3
        else:
            layers = nn.ModuleList([model.module.conv1, model.module.bn1, nn.ReLU(inplace=True), model.module.layer1, model.module.layer2, model.module.layer3])
    elif layer == 4:
        if only:
            layers = model.module.layer4
        else:
            layers = nn.ModuleList([model.module.conv1, model.module.bn1, nn.ReLU(inplace=True), model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4])

    return layers.parameters()

def check_all(project):
    long = len(project)
    i = 0
    for elem in project:
        if elem:
            i += 1
    if i == long:
        return True
    else:
        return False

def single_true(projector):
    i = iter(projector)
    return any(i) and not any(i)

'''-------------------Loss Functions---------------------'''
class Entropy(torch.nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        return -(x.softmax(0)*x.log_softmax(0)).sum(0).mean()

'''--------------------Testing Function-----------------------------'''
def test_batch(net, inputs, labels):
    net.eval()
    with torch.no_grad():
        outputs = net(inputs)
        predicted = torch.argmax(outputs, dim=1)
        correctness = predicted.eq(labels)
    return correctness

'''--------------------Adaptation Function-----------------------------'''
def adapt_batch(net, niter, inputs, opt, K):
    net.train()
    entropy = Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    for iteration in range(niter):
        net.train()
        _, projections = net(inputs, project=True)
        loss = 0.0
        for proj in projections:
            mean_proj = proj.mean(dim=0)
            loss += entropy(proj) + kl(mean_proj.log_softmax(dim=0), torch.full_like(mean_proj, 1 / K))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

