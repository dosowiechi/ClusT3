# ClusT3

Official repository of the ICCV 2023 paper "ClusT3: Information Invariant Test-Time Training", by Gustavo A. Vargas Hakim, David Osowiechi, Mehrdad Noori, Milad Cheraghalikhani, Ali Bahri, Ismail Ben Ayed, and Christian Desrosiers.
The whole article can be found [here](https://arxiv.org/*********).
This work was greatly inspired by the code in [TTTFlow](https://github.com/GustavoVargasHakim/TTTFlow/).

We propose a novel unsupervised TTT technique based on the maximization of Mutual Information between multi-scale feature maps and a discrete latent representation, which can be integrated to the standard training as an auxiliary clustering task.

![Diagram](https://github.com/dosowiechi/ClusT3/blob/master/ClusT3.png)

## Setup 

The repository is divided into two directories: one for the datasets under CIFAR-10 and CIFAR-100 and one for the dataset ViSDA-C. Details of how each code works can be found in their respective READMEs.

## Citation

If you found this repository, or its related paper useful for your research, you can cite this work as:

```
@inproceedings{TTTFlow2023,
  title={ClusT3: Information Invariant Test-Time Training},
  author={Gustavo A. Vargas Hakim and David Osowiechi and Mehrdad Noori  and Milad Cheraghalikhani and Ali Bahri  and Ismail Ben Ayed and Christian Desrosiers},
  booktitle={***},
  pages={},
  month={January},
  year={2023}
}
```
