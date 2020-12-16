# Binarized Neural Architecture Search for Efficient Object Recognition

Neural architecture search (NAS) can have a significant impact in computer vision by automatically designing optimal neural network architectures for various tasks. A variant, binarized neural architecture search (BNAS), with a search space of binarized convolutions, can produce extremely compressed models. Unfortunately, this area remains largely unexplored. BNAS is more challenging than NAS due to the learning inefficiency caused by optimization requirements and the huge architecture space. To address these issues, we introduce channel sampling and operation space reduction into a differentiable NAS to significantly reduce the cost of searching. This is accomplished through a performance-based strategy used to abandon less potential operations. Two optimization methods for binarized neural networks are used to validate the effectiveness of our BNAS. Extensive experiments demonstrate that the proposed BNAS achieves a performance comparable to NAS on both CIFAR and ImageNet databases. An accuracy of 96.53% vs. 97.22% is achieved on the CIFAR-10 dataset, but with a significantly compressed model, and a 40% faster search than the state-of-the-art PC-DARTS.

Here we provide our test codes and pretrained models.

https://arxiv.org/abs/1911.10862

## Requirements

python 3.6

PyTorch 1.0.0

## Run examples

You need to modified your path to dataset using --data_path_cifar.

To evaluate the model in CIFAR-10, just run

```bash
sh script/cifar10_test_xnor_larger.sh
```
