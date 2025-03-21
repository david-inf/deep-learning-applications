# Deep learning applications

Repository to host the laboratories from the course on deep learning applications

## Lab1 - Convolutional Neural Networks

Feel for working with deep models

### Code organization

Inside folder `lab1/` you have the main script `lab1.py` see
```{bash}
python lab1.py --help
```

You should see
```{bash}
030: 100%|██████████████████████| 338/338 [00:02<00:00, 112.93batch/s, train_acc=0.997, train_loss=0.0669, val_acc=0.538, val_loss=1.97]
031: 100%|██████████████████████| 338/338 [00:03<00:00, 112.49batch/s, train_acc=0.998, train_loss=0.0619, val_acc=0.533, val_loss=1.97]
032: 100%|██████████████████████| 338/338 [00:03<00:00, 106.86batch/s, train_acc=0.998, train_loss=0.0583, val_acc=0.534, val_loss=1.98]
```

After generating configs with `generate_config`, run a program with
```{bash}
python lab1.py experiments/CNN_4.83M_cifar10.yaml
```

It will automatically save checkpoints and log to `comet_ml`. If the experiment have already been runned, you may run the same command with more epochs (`--epochs 40`) and the experiment will be resumed (checkpoint path and experiment key are automatically dumped in the configuration file).

- `models/` folder with `mlp.py` and `cnn.py`
- `config.yaml` base configuration file
- `mydata.py` wrappers for MNIST and CIFAR10 datasets
- `train.py` and `utils.py` are utilities

### Exercise 1

Reproducing on a small scale the results from the ResNet paper, using MNIST and CIFAR10 datasets, an MLP and a CNN.

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

Reproducing on a small scale the results from the distillation paper, using MNIST and CIFAR10 datasets, an MLP and a CNN.

> Distilling the Knowledge in a Neural Network, Geoffrey Hinton, Oriol Vinyals, Jeff Dean. [https://arxiv.org/abs/1503.02531].

For a given $x$ the frozen teacher and trainable students both produce logits, the idea is to align the student's output with the teachers' one

Teachers:
- 
Students:
- 

Loss:
- Soft targets loss: `KLDivLoss(log_target=True)(soft_prob, soft_targets)`
- Hard targets loss: `CrossEntropyLoss()(student_logits, labels)`
</details>
