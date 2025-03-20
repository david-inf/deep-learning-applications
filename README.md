# Deep learning applications

Repository to host the laboratories from the course on deep learning applications

## Lab1 - Convolutional Neural Networks

Feel for working with deep models

### Exercise 1

Reproducing on a small scale the results from the ResNet paper, using MNIST dataset, an MLP and the a CNN.

> Deep Residual Learning for Image Recognition, Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, 2015. [https://arxiv.org/abs/1512.03385].

> Deeper networks do not guarantee more reduction in training loss

<details>
<summary>Baseline MLP `BaseMLP`</summary>

MLP with variable number of blocks:
- `BasicBlock`: 2 fully connected layers with `hidden_size=512` and relu
- Optional skip connection in each block by setting `skip=True`

Datasets:
- `MNIST`: 
- `CIFAR10`: 
</details>

<details>
<summary>Baseline CNN `BaseCNN`</summary>

- `input_adapter`: conv + batchnorm + relu that exits with `num_filters`
- `layer`: sequence of `BasicBlock` layers
    - Two modules of conv + batchnorm + relu
    - Optional shortcut in each block by setting `skip=True`
- `avgpool`: ends with a (1, 1) feature map
- `fc`: classification head

Datasets:
- `MNIST`: 4,800,000 params
- `CIFAR10`: 4,800,000 params
</details>

**Experiment** | **Results**
-------------- | -----------
description | plot

### Exercise 2

<details>
<summary>Distillation</summary>


</details>
