# ClusT3 on the CIFAR datasets

### Datasets

The experiments utilize the CIFAR-10 training split as the source dataset. It can be downloaded from 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), or can also be done using torchvision
datasets: `train_data = torchvision.datasets.CIFAR10(root='Your/Path/To/Data', train=True, download=True)`.
The same line of code can be used to load the data if it is already downloaded, just by changing the
argument `download` to `False`.

At test-time, we use CIFAR-10-C and CIFAR-10-new. The first one can be downloaded from [CIFAR-10-C](
https://zenodo.org/record/2535967#.YzHFMXbMJPY). For the second one, please download the files 
`cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy` from [CIFAR-10-new](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets).
All the data should be placed in a common folder from which they can be loaded, e.g., `/datasets/`.

The training works the same way on CIFAR-100 dataset and it can be downloaded from [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).
At test-time, we use CIFAR-100-C which can be downloaded from [CIFAR-100-C](https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1).

## Source Training

Do not forget to change the specified roots in `configuration.py`.

The file `MultiGPUs_training.py` can be used for training the architecture of ClusT3 with ResNet-50 as baseline. To modify the architecture of ClusT3, please edit the `configuration.py`, setting `--project-layerX` to 1 if you want it to be on layer X or 0 otherwise with X between 1 and 4, changing the number of cluster `--K` or the number of projectors per layer `--multi`. The best configuration in this case is `--project-layer1 1 --project-layer2 1 --K 10 --multi 15`.

## Test-time Adaptation

At test-time, we utilize `test_adapt_projector.py` to adapt our model which should be defined as in source training using `configuration.py`.
Depending on the `--niter` choosen between [1, 3, 5, 10, 20, 50, 100], we obtain the accuracy for each number of iterations from the list up to the `--niter`..
