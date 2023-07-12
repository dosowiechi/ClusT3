import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision

def save_checkpoint(state, is_best, args):
    torch.save(state, args.save + args.dataset + '_' + args.model + '.pth')
    if is_best:
            torch.save(state, args.save + args.dataset + '_' + args.model + '_torch_best.pth')

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

'''--------------------Adaptation Function-----------------------------'''
def adapt_batch(net, niter, inputs, opt, K, iterations, save_iter, projection, proj_layers):
    net.inference = False
    net.train()
    entropy = Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    for iteration in range(niter):
        if projection:
            _, out_proj = net(inputs, feature=False, projection=projection)
            loss = 0
            for i, x in enumerate(out_proj):
                x_mean = x.mean(dim=1)
                loss += entropy(x) + kl(F.log_softmax(x_mean, dim=0), torch.full_like(x_mean, 1 / K))
                # loss += -((x.softmax(1) + 1 / K) * x.log_softmax(1)).sum(1).mean()  # Entropy + KL with uniform
        else:
            x = net(inputs, feature=False, projection=projection)
            x_mean = x.mean(dim=0)
            loss += entropy(x) + kl(F.log_softmax(x_mean, dim=0), torch.full_like(x_mean, 1 / K))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if iteration+1 in iterations:
            weights = {'weights': net.state_dict()}
            torch.save(weights, save_iter + 'weights_iter_' + str(iteration+1) +'.pkl')
    net.eval()


def adapt_batch_entropy(net, niter, inputs, opt, K, iterations, save_iter, projection, proj_layers):
    net.inference = False
    net.train()
    entropy = Entropy()
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    for iteration in range(niter):
        if projection:
            _, out_proj = net(inputs, feature=False, projection=projection)
            loss = 0
            for i, layer in enumerate(proj_layers):
                if layer != None :
                    x = out_proj[i]
                    loss += entropy(x)
                    # loss += -((x.softmax(1) + 1 / K) * x.log_softmax(1)).sum(1).mean()  # Entropy + KL with uniform
        else:
            x = net(inputs, feature=False, projection=projection)
            x_mean = x.mean(dim=0)
            loss = entropy(x) + kl(F.log_softmax(x_mean, dim=0), torch.full_like(x_mean, 1 / K))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if iteration+1 in iterations:
            weights = {'weights': net.state_dict()}
            torch.save(weights, save_iter + 'weights_iter_' + str(iteration+1) +'.pkl')
    net.eval()


'''--------------------Testing Function-----------------------------'''
def test_batch(net, inputs, labels, adapt=False):
    net.eval()
    net.inference = False
    with torch.no_grad():
        outputs = net(inputs)
        acc = accuracy(outputs, labels)
        predicted = torch.argmax(outputs, dim=1)
        correctness = predicted.eq(labels).cpu()
    return correctness, acc

'''-------------------Loss Functions----------------------------------'''
class Entropy(torch.nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        return -(x.softmax(0)*x.log_softmax(0)).sum(0).mean()

'''-------------------Getting Adapters Parameters---------------------'''
def get_parameters(layers, model):
    parameters = []
    if layers[0] is not None:
        parameters += list(model.mask1.parameters())
    if layers[1] is not None:
        parameters += list(model.mask2.parameters())
    if layers[2] is not None:
        parameters += list(model.mask3.parameters())
    if layers[3] is not None:
        parameters += list(model.mask4.parameters())
    return parameters

def extractor_from_layer2(net):
    layers = [net.conv1,  net.bn1, nn.ReLU(inplace=True), net.layer1, net.layer2]
    return nn.Sequential(*layers)

def neg_log_likelihood_2d(target, z, log_det):
    log_likelihood_per_dim = target.log_prob(z) + log_det
    return -log_likelihood_per_dim.mean()

def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E
