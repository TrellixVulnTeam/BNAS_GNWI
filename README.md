# Binarized Neural Architecture Search for Efficient Object Recognition

Traditional neural architecture search (NAS) has a significant im- pact in computer vision by automatically designing network architectures for various tasks. In this paper, binarized neural architecture search (BNAS), with a search space of binarized convolutions, is introduced to produce extremely compressed models to reduce huge computational cost on embedded devices for edge computing. The BNAS calculation is more challenging than NAS due to the learning inefficiency caused by optimization requirements and the huge architecture space, and the performance loss when handling the wild data in various computing applications. To address these issues, we introduce opera- tion space reduction and channel sampling into BNAS to significantly reduce the cost of searching. This is accomplished through a performance-based strat- egy that is robust to wild data, which is further used to abandon less potential operations. Furthermore, we introduce the Upper Confidence Bound (UCB) to solve 1-bit BNAS. Two optimization methods for binarized neural networks are used to validate the effectiveness of our BNAS. Extensive experiments demonstrate that the proposed BNAS achieves a comparable performance to NAS on both CIFAR and ImageNet databases. An accuracy of 96.53% vs. 97.22% is achieved on the CIFAR-10 dataset, but with a significantly com- pressed model, and a 40% faster search than the state-of-the-art PC-DARTS.

Here we provide our test codes and pretrained models.

https://arxiv.org/pdf/2009.04247.pdf

## Requirements

python 3.6
PyTorch 1.0.0

## Run examples

You need to modified your path to dataset using --data_path_cifar.

To evaluate the model in CIFAR-10, just run

```bash
sh script/cifar10_test_xnor_larger.sh
```
