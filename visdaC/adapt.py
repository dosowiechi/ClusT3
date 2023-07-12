import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
from utils import utils, create_model, prepare_dataset

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    '''--------------------Loading Model-----------------------------'''
    projectors = tuple(map(bool, args.layers))
    print('Loading model')
    num_classes = 12
    ckpt = args.root + '/weights/' + args.dataset + '_' + utils.list_to_str(args.layers) + '_W' + str(args.gamma) + '_N' + str(args.multi) + '_S' + args.psize + '_K' + str(args.K) + '.pth'
    weights = torch.load(ckpt)['state_dict']
    model = create_model.create_model(args, projectors=projectors).cuda()
    model.load_state_dict(weights)

    '''--------------------Getting Parameters-----------------------------'''
    parameters = utils.get_parameters(mode='original', project=projectors, model=model)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.plr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.plr)

    '''-------------------Loading Dataset----------------------------'''
    if args.domain == 'test':
        teloader, _ = prepare_dataset.prepare_test_data(args)
    else:
        teloader, _ = prepare_dataset.prepare_val_data(args)

    '''--------------------Test-Time Adaptation----------------------'''
    print('Test-Time Adaptation')
    good_good = []
    good_bad = []
    bad_good = []
    bad_bad = []
    correct = 0
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        model.load_state_dict(weights)
        correctness = utils.test_batch(model, inputs, labels)

        if args.adapt:
            utils.adapt_batch(model, args.niter, inputs, optimizer, args.K)
            correctness_new = utils.test_batch(model, inputs, labels)
            for i in range(len(correctness_new.tolist())):
                if correctness[i] == True and correctness_new[i] == True:
                    good_good.append(1)
                elif correctness[i] == True and correctness_new[i] == False:
                    good_bad.append(1)
                elif correctness[i] == False and correctness_new[i] == True:
                    bad_good.append(1)
                elif correctness[i] == False and correctness_new[i] == False:
                    bad_bad.append(1)
        else:
            correct += correctness.sum().item()

    correct += len(good_good) + len(bad_good)
    accuracy = correct / len(teloader.dataset)
    print('--------------------RESULTS----------------------')
    print('Perturbation: ', args.corruption)
    print('Accuracy: ', accuracy)
    if args.adapt:
        print('Good first, good after: ', len(good_good))
        print('Good first, bad after: ', len(good_bad))
        print('Bad first, good after: ', len(bad_good))
        print('Bad first, bad after: ', len(bad_bad))


if __name__ == '__main__':
    args = configuration.argparser()
    if args.livia:
        args.dataroot = '/export/livia/home/vision/gvargas/data/'
        args.root = '/export/livia/home/vision/gvargas/MaskUp'

    experiment(args)