# Deep learning applications

Repository to host the laboratories from the course on Deep Learning Applications. We conver topics ranging from Computer Vision, Natural Language Processing and then Adversarial Learning.


## :test_tube: Lab1 - Convolutional Neural Networks

Feel for working with deep models

Inspect experiments from my [comet_ml](https://www.comet.com/david-inf/deep-learning-applications) project

<details>
<summary>Code organization</summary>

```bash
pip install -r lab1.txt
```

- `lab1/ckpts/` folder that will be automatically created for storing model checkpoints, this uses `torch.save()`
- `lab1/configs/` folder that will be automatically created for storing `yaml` configurations files for each experiment
  - `generate_configs.py` automatically generate a configuration file from a given params dict
  - Each model configuration will be stored in `configs/model/`
- `lab1/models/` module with MLPs (`mlp.py`) and CNNs (`cnn.py` `resnet.py` `wideresnet.py`) definitions
- `lab1/plots/` for results
- `lab1/utils/` module with utilities (`misc.py` and `train.py`)
- `lab1/cmd_args.py` arguments for main programs
- `lab1/mydata.py` wrappers for MNIST and CIFAR10 datasets, augmentations are available too
- `lab1/train.py` `lab1/distill.py` training utilities for standard training and knowledge distillation training
- Main programs:
  - `lab1/main_train.py` main script for training a single model, see `python lab1/main_train.py --help`
  - `lab1/main_distill.py` main script for distilling knowledge, see `python lab1/main_distill.py --help`

Run the full lab with these two shell scripts

```bash
chmod +x ./lab1/commands1.sh
./lab1/commands1.sh
```

```bash
chmod +x ./lab1/commands2.sh
./lab1/commands2.sh
```

</details>

<details>
<summary>Running the main programs</summary>

Before running check always if the configuration file is correct (as for the device).

```bash
python lab1/main_train.py --config lab1/configs/CNN/MediumCNN.yaml --view
```

```bash
python lab1/main_train.py --config lab1/configs/CNN/MediumCNN.yaml
```

```bash
001: 100%|█████████████████████████| 391/391 [00:30<00:00, 12.92batch/s, train_acc=0.342, train_loss=1.73, val_acc=0.379, val_loss=1.78]
002: 100%|█████████████████████████| 391/391 [00:37<00:00, 10.32batch/s, train_acc=0.5, train_loss=1.37, val_acc=0.535, val_loss=1.28]
003: 100%|█████████████████████████| 391/391 [00:39<00:00,  9.91batch/s, train_acc=0.586, train_loss=1.15, val_acc=0.597, val_loss=1.16]
```

```bash
python lab1/main_distil.py --config lab1/configs/Distil/DistilCNN_RN32.yaml
```

```bash
001: 100%|████████████████████████| 391/391 [00:13<00:00, 28.11batch/s, train_acc=0.326, train_loss=2.32, val_acc=0.413, val_loss=1.62]
002: 100%|████████████████████████| 391/391 [00:12<00:00, 31.35batch/s, train_acc=0.472, train_loss=1.74, val_acc=0.497, val_loss=1.49]
003: 100%|████████████████████████| 391/391 [00:12<00:00, 31.09batch/s, train_acc=0.537, train_loss=1.48, val_acc=0.55, val_loss=1.27]
```

</details>


### :zero: Warming up with MNIST

Train a MLP and a two CNNs on the MNIST dataset. I chose to train two CNNs because one has fewer params than the dataset samples, the other has more, as the MLP. Maybe something shows up idk.

<details>
<summary>MLP architecture</summary>

The simplest version in which you give as argument a list with hidden unit sizes `layer_sizes=[512, 512, 512]` like in this example. On top of this another linear layer that ends with the number of classes.

```python
layers = []
layers.append(nn.Linear(input_size, layer_sizes[0]))
layers.append(nn.ReLU(inplace=True))
for i in range(len(layer_sizes) - 1):
    layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    layers.append(nn.ReLU(inplace=True))
self.mlp = nn.Sequential(*layers)

self.classifier = nn.Linear(layer_sizes[-1], num_classes)
```

- `python lab1/main_train.py --config lab1/configs/MLP/MLP_mnist.yaml --view`

</details>

<details>
<summary>CNNs architecture</summary>

This architecture follows the concept of the ResNet in which we have "macro-layers" each one with a variable number of blocks.

- `input_adapter`: conv + batchnorm + relu that exits with `num_filters`
- `blocks`: fixed number (2) of layers with variable `BasicBlock` blocks
  - Each `BasicBlock` contains two modules of conv + batchnorm + relu
  - Each layer contains $n$ `BasicBlock`, in the default version $n=1$ (this is specified via the `num_blocks` argument)
  - Optional skip connection in each block by setting `skip=True` (for residual learning comparison)
- `avgpool`: ends with a `(num_filters*2) x 1 x 1` feature map
- `classifier`: classification head

Here we use 2 macro-layers, resulting in `2*2*n+2` total layers.

- `python lab1/main_train.py --config lab1/configs/CNN/CNN1_mnist.yaml --view` where `num_blocks=2` and `num_filters=32`
- `python lab1/main_train.py --config lab1/configs/CNN/CNN2_mnist.yaml --view` where `num_blocks=2` and `num_filters=64`

</details>

<details>
<summary>Results</summary>

| Model  | #params | val_ac |
| ------ | ------- | ------ |
| `MLP`  | 0.93M   | 0.9866 |
| `CNN1` | 0.17M   | 0.9956 |
| `CNN2` | 0.68M   | 0.9958 |

- `python lab1/main_train.py --config lab1/configs/MLP/MLP_mnist.yaml`
- `python lab1/main_train.py --config lab1/configs/CNN/CNN1_mnist.yaml`
- `python lab1/main_train.py --config lab1/configs/CNN/CNN2_mnist.yaml`

<p align="middle">
  <img src="lab1/plots/mnist_warmup.svg" alt="Warming up on MNIST" width="60%">
</p>

</details>


### :one: Degradation problem, deep residual learning

Reproducing on a small scale the results from the ResNet paper on CIFAR10 dataset.

> Deep Residual Learning for Image Recognition. He *et al*. [Arxiv](https://arxiv.org/abs/1512.03385).

Deeper networks, i.e. more stacked layers, do not guarantee more reduction in training loss. So the point of this exercise is to abstract a model definition so that one can add a given number of layers (blocks), and then see how the performance are affected. The idea is to reproduce Figure 6 from the paper.

<details>
<summary>Results</summary>

| Model           | `num_blocks` | `num_filters` | #params | Layers | val_acc |
| --------------- | ------------ | ------------- | ------- | ------ | ------- |
| `SmallCNN`      | 1            | 16            | 0.02M   | 6      | 0.7091  |
| `SmallCNNskip`  | 1            | 16            | 0.02M   | 6      | 0.6891  |
| `MediumCNN`     | 5            | 16            | 0.11M   | 22     | 0.7418  |
| `MediumCNNskip` | 5            | 16            | 0.11M   | 22     | 0.7975  |
| `LargeCNN`      | 7            | 16            | 0.16M   | 30     | 0.6916  |
| `LargeCNNskip`  | 7            | 16            | 0.16M   | 30     | 0.8034  |

<p align="middle">
  <img src="lab1/plots/deg_prob.svg" alt="learning" width="60%">
</p>

When adding further layers we see that "adding more layers reduces loss" holds no more. Skip connections, residual learning, solve the problem. Validation accuracy provides evidence as well, i.e. skip connections solve the degradation problem.

</details>


### :two: Improving over ResNet with WideResNet

Now we need to find our flagship model that will be used later for knowledge distillation. We take a step forward from the previous CNNs by adding one more macro-layer as in the ResNet paper, and we do a comparison with another model, namely the WideResNet.

> Wide Residual Networks. Zagoruyko *et al*. [Axiv](https://arxiv.org/abs/1605.07146).

<details>
<summary>RN and WRN architectures</summary>

As said the RN is the same as the previous CNN with one more macro-layer

- `input_adapter`: conv + batchnorm + relu that exits with `num_filters`
- `blocks`: fixed number (3) of layers with variable `BasicBlock` blocks
  - Each `BasicBlock` contains two modules of conv + batchnorm + relu
  - Each layer contains $n$ `BasicBlock`, in the default version $n=1$ (this is specified via the `num_blocks` argument)
  - Optional skip connection in each block by setting `skip=True` (for residual learning comparison)
- `avgpool`: ends with a `(num_filters*2) x 1 x 1` feature map
- `classifier`: classification head

The WRN has the same general architecture except for the `BasicBlock` where we have the so called pre-activation, that is batchnorm + relu + conv. More than this, the *wide* lies the `k` parameter that multiplies the `num_filter` parameter results in `k` more filters than the RN in each layers. This allows to grow the network in width rather than in depth.

</details>

<details>
<summary>Results</summary>

| Name             | `num_blocks` | `num_filters` | `widen_factor` | #params | Layers | val_acc |
| ---------------- | ------------ | ------------- | -------------- | ------- | ------ | ------- |
| `ResNet32`       | 5            | 16            | 1              | 0.47M   | 32     | 0.8301  |
| `ResNet44`       | 7            | 16            | 1              | 0.66M   | 44     | 0.8441  |
| `ResNet56`       | 9            | 16            | 1              | 0.86M   | 56     | 0.8339  |
| `WideResNet14-2` | 2            | 16            | 2              | 0.69M   | 14     | 0.8549  |
| `WideresNet14-4` | 2            | 16            | 4              | 2.75M   | 14     | 0.8789  |
| `WideResNet26-4` | 4            | 16            | 4              | 5.85M   | 26     | 0.8858  |


<p align="middle">
  <img src="lab1/plots/rn_wrn.svg" alt="WRN vs RN", width="60%">
</p>

</details>


### :three: Knowledge Distillation

Reproducing on a small scale the results from the distillation paper on CIFAR10 dataset. Having two flagship models (), since we want to one more comparison between Rn and WRN.

> Distilling the Knowledge in a Neural Network. Hinton *et al*. [Arxiv](https://arxiv.org/abs/1503.02531).

<details>
<summary>Learning algorithm</summary>

For a given $x$ the frozen teacher and the trainable students both produce logits, the idea is to align the
student's output with the teacher's one.

Loss:
- Soft targets loss $\mathcal{L}_1$: `KLDivLoss(log_target=True, reduction="batchmean")(soft_prob, soft_targets)`
- Hard targets loss $\mathcal{L}_2$: `CrossEntropyLoss()(student_logits, labels)`
- Final loss: $\mathcal{L}=w_1\mathcal{L}_1+w_2\mathcal{L}_2$ with $w_1\gg w_2$ which important to ensure that the knowledge distillation training outperforms that standard training

As the teacher model we use the actual `ResNet` architecture with 3 blocks of `BasicBlock` blocks resulting in
$3n+2$ total layers. Also the same algorithm is applied to the `WideResNet` model (same architecture with pre-activation `BasicBlock`).

</details>

<details>
<summary>Results</summary>

We define another CNN, named BaseCNN, with skip connections and to have more #params than dataset samples. Here we'd like to compare BaseCNN with standard training and knowledge distillation training. We compare also the two teachers (also warly stopping was applied).

- `python lab1/main_distil.py --config lab1/configs/ResNet/ResNet56.yaml`
- `python lab1/main_distil.py --config lab1/configs/WideResNet/WideResNet26-4.yaml`

| Name                 | `num_blocks` | `num_filters` | `widen_factor` | #params | Layers | val_acc |
| -------------------- | ------------ | ------------- | -------------- | ------- | ------ | ------- |
| `ResNet56`           | 9            | 16            | 1              | 0.86M   | 56     | 0.8339  |
| `WideResNet26-4`     | 4            | 16            | 4              | 5.85M   | 26     | 0.8858  |
| `BaseCNN`            | 1            | 32            | 1              | 0.08M   | 6      | 0.7608  |
| `DistilCNN_RN56`     | 1            | 32            | 1              | 0.08M   | 6      | 0.7720  |
| `DistilCNN_WRN26-4`  | 1            | 32            | 1              | 0.08M   | 6      | 0.6742  |

<p align="middle">
  <img src="lab1/plots/distil.svg" alt="learning" width="60%">
</p>

The distilled model is able to achieve a higher train accuracy earlier. Mostly similar performance on the validation set, however the distilled model stays on top of the base one. The small model trained with distillation has better performance than the same trained in the classical way! And WideResNet outperforms ResNet on both comparisons.

</details>


## :test_tube: Lab3 - Transformers and NLP

Work with the `HuggingFace` ecosystem to adapt models to new tasks.

<details>
<summary>Code organization</summary>

```bash
python install -r lab3.txt
```

- `lab3/ckpts/` model checkpoints using `.save_pretrained()` method
- `lab3/configs/` configuration files automatically generated using `generate_configs.py` program
- `lab3/models/` wrappers for BERT-family models
- `lab3/results/` plotted stuffs
- `lab3/utils/` module with various utilities inside `misc.py` and `train.py`
- `lab3/cmd_args.py` main programs' arguments
- `lab3/load_and_eval.py` load the validation or testsplits  and perform inference with a given model checkpoint (you must train first)
- `lab3/main_extract.py` main program for obtaining baseline results with a given pretrained extractor, i.e. a BERT-family model from the local `models` module
- `lab3/main_ft.py` core of this lab that is the main program for finetuning a pretrained BERT-family model given a configuration file
- `lab3/mydata.py` utilities for preprocessing and loading the `rotten_tomatoes` dataset from HuggingFace
- `lab3/train.py` train loop

Try `python lab3/main_extract.py --help` and `python lab3/main_ft.py --help`. You'll see that for `main_ft.py` there's the `--view` argument available, that allows to inspect a model given its configuration file via the `--config` argument.

</details>


### :one: BERT as a feature extractor

Train a simple classifier (LinearSVC) on top of BERT sentence representation for sentiment analysis task, this will be the baseline which we will try to improve with finetuning. See code in `lab3/main_extract.py`.


<details>
<summary>Results</summary>

We use the rotten tomatoes dataset with train-val-test splits, hence we use the BERT-family models as feature extractors, then we train a LinearSVC classifier on top of the representation. We compare DistilBERT (`[CLS]` token and mean pooling) and SentenceBERT (two models) extractors.

```bash
chmod +x ./lab3/baseline.sh
./lab3/baseline.sh
```

The `--extract` argument is needed for saving the features locally, this makes possible to train different classifiers on top of those features.

| Extractor for LinearSVC                   | size  | `train_acc` | `val_acc` |
| ----------------------------------------  | ----- | ----------- | --------- |
| `distilbert-base-uncased` (`[CLS]` token) | 67M   | 0.849       | 0.822     |
| `distilbert-base-uncased` (mean pooling)  | 67M   | 0.846       | 0.810     |
| `sentence-transformers/all-MiniLM-L6-v2`  | 22.7M | 0.791       | 0.767     |
| `sentence-transformers/all-mpnet-base-v2` | 109M  | 0.879       | 0.855     |

Being SBERT more suitable than DistilBERT for producing sentence embeddings, as we expected the classifier on top of SBERT has better performance.

</details>


<details>
<summary>Visualize embeddings</summary>

Here we load the features extracted before and visualize them in a 2D space using the UMAP method, see `view_embeds()` function from `lab3/main_extract.py`.-

```bash
python lab3/main_extract.py --extractor distilbert --method cls  --view
python lab3/main_extract.py --extractor sbert --method mpnet --view
```

<p align="middle">
  <img src="lab3/results/distilbert_embeds.svg" alt="Pretrained DistilBERT embeddings" width="45%">
  &nbsp;
  <img src="lab3/results/sbert_embeds.svg" alt="Best pretrained SentenceBERT embeddings" width="45%">
</p>

Here we see the expressive power of SBERT against DistilBERT :)

</details>


### :two: BERT Finetuning

The goal now is to improve over the baseline performance. For doing this we proceed with a full finetuning and see what happens. Then we seek for a more efficient way for finetuning BERT on the rotten tomatoes dataset using `PEFT` library. See code in `lab3/main_ft.py`.

The idea is to perform model selection on BERT-family models (full-finetuning and few LoRA configs) for the text classification task, then we deploy the best BERT on the test split. So I'd like to reproduce results from figure 2 of the original LoRA paper

> LoRA: Low-Rank Adaptation of Large Language Models. Hu *et al*. [Arxiv](https://arxiv.org/abs/2106.09685).

<details>
<summary>Finetuning settings</summary>

So we compare the full-finetuning and few LoRA configurations, for defining these configurations we use the following matrix, since the two questions are (i) do we need to update all the parameters? (ii) how expressive should the updates be?

| params/rank | 8            | 16            |
| ----------- | ------------ | ------------- |
| **q**       | `lora_q8`    | `lora_q16`    |
| **qv**      | `lora_qv8`   | `lora_qv16`   |
| **qkvo**    | `lora_qkvo8` | `lora_qkvo16` |

```bash
chmod +x ./lab3/finetuning.sh
./lab3/finetuning.sh
```

| Model                    | #params (log10) | val_acc |
| ------------------------ | --------------- | ------- |
| `distilbert_full`        | 7.83            | 0.860   |
| `distilbert_lora_q8`     | 5.82            | 0.853   |
| `distilbert_lora_q16`    | 5.87            | 0.853   |
| `distilbert_lora_qv8`    | 5.87            | 0.839   |
| `distilbert_lora_qv16`   | 5.95            | 0.845   |
| `distilbert_lora_qkvo8`  | 5.95            | 0.848   |
| `distilbert_lora_qkvo16` | 6.07            | 0.850   |

Being the full-finetuning not that much expensive to train, this will be the final model to deploy on unseen data.

</details>


<details>
<summary>LoRA and full-finetuning</summary>

Efficient way for finetuning BERT on rotten tomatoes dataset using `PEFT` library

<p align="middle">
  <img src="lab3/results/lora.svg" alt="LoRA against full-finetuning" width="50%">
</p>

</details>


<details>
<summary>Deploy on onseen data</summary>

Obviously the full-finetuned DistilBERT has the better performance, and since the finetuning isn't that much expensive yet, `distilbert_full` will be deployed on unseed data from rotten tomatoes dataset, i.e. the test split.

```bash
python lab3/load_and_eval.py --split test --config lab3/configs/distilbert_full.yaml
```

This results in an accuracy value of `0.841`.

</details>


## :test_tube: Lab4 - Adversarial Learning

<details>
<summary>Code organization</summary>

```bash
pip install -r lab1.txt
```

Inside the `lab4/` folder there are the following programs

- `lab4/ckpts/` checkpoints and configuration files
- `lab4/models/` with `autoencoder.py`
- `lab4/plots/`
  - Results from OOD detection on CIFAR100 subsets (aquatic mammals and people) and FakeData
- `lab4/utils/` various utilities
- `lab4/main_adversarial.py` main program for experimenting with adversarial attacks
- `lab4/main_detection.py` main program for launching the OOD detection pipeline on the given dataset from `lab4/mydata.py`
- `lab4/main_robust.py` launch a training with adversarial augmentations, see configs in `lab4/ckpts/`
- `lab4/mydata.py` various datasets for OOD detection
- `lab4/train_ae.py` main program for training the AutoEncoder on CIFAR10 dataset
- `lab4/train_ccn.py` utilities for the adversarial training

</details>


### :one: Adversarial attacks

Now we move to adversarial attacks by visualizing what these attacks are about

<details>
<summary>Attacks</summary>

Run the shell script that contains commands for running an untargeted and targeted attacks

```bash
chmod +x ./lab4/adversarials.sh
./lab4/adversarials.sh
```

<table>
  <caption>Targeted and untarged attacks
  <tr>
    <td><img src="lab4/plots/adversarial/untargeted1.svg"></td>
    <td><img src="lab4/plots/adversarial/targeted1.svg"></td>
  </tr>
</table>

</details>

<details>
<summary>More attacks</summary>

<table>
  <caption>Targeted and untarged attacks
  <tr>
    <td><img src="lab4/plots/adversarial/untargeted2.svg"></td>
    <td><img src="lab4/plots/adversarial/targeted2.svg"></td>
  </tr>
</table>

</details>


### :two: Enhancing robustness to adversarial attacks

We take a base model and enhance its robustness to these adversarial attacks. The idea is to train again this base model on a dataset that is augmented with untargeted adversarial attacks. By doing this the base model robustness to adversarial attacks, and will also be reflected in the ability to detect OOD examples.

<details>
<summary>Results</summary>

Inspect model via `python lab4/main_robust.py --config lab4/ckpts/cnn_robust.yaml --view` then launch training

```bash
python lab4/main_robust.py --config lab4/ckpts/cnn_robust.yaml
```

This script runs the full pipeline that comprises the adversarial training to obtain the model `RobustCNN` then the OOD detection pipeline. The results are already displayed in the OOD detection pipeline.

Launch this script to evaluate the model on the CIFAR10 test split `python lab1/load_and_eval.py --config lab4/ckpts/cnn_robust.yaml` you will see the accuracy is at `0.662`. So, yeah the model might be robust to adversarial attacks, bu the accuracy is very low, comapared to its standard version `python lab1/load_and_eval.py --config lab1/configs/CNN/LargeCNNskip.yaml` at `0.798`.

<table>
  <caption>Targeted and untarged attacks
  <tr>
    <td><img src="lab4/plots/adversarial/untargeted_rcnn.svg"></td>
    <td><img src="lab4/plots/adversarial/targeted_rcnn.svg"></td>
  </tr>
</table>

</details>


### :three: OOD detection pipeline

<details>
<summary>ID and OOD samples</summary>

We choose as in-distribution (ID) dataset CIFAR10 (10000 samples from test split), and few out-of-distribution (OOD) datasets
- **aquatic mammals** subset from CIFAR100 (2500 samples from train split) `python lab4/mydata --ood aquatic`
- **people** subset from CIFAR100 `python lab4/mydata --ood people`
- **noise** generate from FakeData dataset `python lab4/mydata --ood noise`

<table>
  <caption>CIFAR10, CIFAR100 (aquatic mammals), CIFAR100 (people), and FakeData
  <tr>
    <td><img src="lab4/plots/id_imgs.png" alt="ID samples" width="100%"></td>
    <td><img src="lab4/plots/aquatic/ood_imgs.png" alt="OOD samples" width="100%"></td>
    <td><img src="lab4/plots/people/ood_imgs.png" alt="OOD samples" width="100%"></td>
    <td><img src="lab4/plots/noise/ood_imgs.png" alt="OOD samples" width="100%"></td>
  </tr>
</table>

</details>

<details>
<summary>AutoEncoder</summary>

Inspect model via `python lab4/train_ae.py --config lab4/ckpts/autoencoder.yaml --view` then launch training

```bash
python lab4/train_ae.py --config lab4/ckpts/autoencoder.yaml
```

This autoencoder is trained to reconstruct ID samples, so when passing an OOD sample, the MSE computes like a distance from its ID version, hence higher the MSE, higher the chance of being OOD - this will be the metric for detecting OOD samples.

The AE outputs with a sigmoid, so images needs to be in [0,1] already, as done in the `lab1/` exercises.

</details>

<details>
<summary>OOD detection pipeline</summary>

OOD detection pipeline for all the OOD datasets chosen, see `python lab4/main_detection.py --help`, plot data with `python lab4/mydata.py`. Do this by changing the code in `lab4/mydata.py` default: FakeData since is the only one dataset in which the AutoEncoder seems to work well. I would say that the method doesn't work on the two CIFAR100 subsets since CIFAR10 is a subset as well, and the distribution might be the same regardless of being different classes.

```bash
chmod +x ./lab4/detection_pipeline.sh
./lab4/detection_pipeline.sh
```

<table>
  <caption>Performance on CIFAR100 aquatic mammals subset</caption>
  <tr>
    <!-- <td><img src="lab4/plots/aquatic/scores_max_logit_LargeCNNskip.svg" alt="Scores from CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/aquatic/scores_max_softmax_LargeCNNskip.svg" alt="Scores from CNN using max_softmax", width="100%"></td>
    <td><img src="lab4/plots/aquatic/scores_mse_AutoEncoder.svg" alt="Scores from CNN using max_logit", width="100%"></td>
    <!-- <td><img src="lab4/plots/aquatic/scores_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/aquatic/scores_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
  <tr>
    <!-- <td><img src="lab4/plots/aquatic/roc_pr_max_logit_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td> -->
    <td><img src="lab4/plots/aquatic/roc_pr_max_softmax_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td>
    <td><img src="lab4/plots/aquatic/roc_pr_mse_AutoEncoder.svg" alt="ROC and PR curves", width="100%"></td>
    <!-- <td><img src="lab4/plots/aquatic/roc_pr_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/aquatic/roc_pr_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
</table>

<table>
  <caption>Performance on CIFAR100 people subset</caption>
  <tr>
    <!-- <td><img src="lab4/plots/people/scores_max_logit_LargeCNNskip.svg" alt="Scores from CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/people/scores_max_softmax_LargeCNNskip.svg" alt="Scores from CNN using max_softmax", width="100%"></td>
    <td><img src="lab4/plots/people/scores_mse_AutoEncoder.svg" alt="Scores from CNN using max_logit", width="100%"></td>
    <!-- <td><img src="lab4/plots/people/scores_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/people/scores_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
  <tr>
    <!-- <td><img src="lab4/plots/people/roc_pr_max_logit_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td> -->
    <td><img src="lab4/plots/people/roc_pr_max_softmax_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td>
    <td><img src="lab4/plots/people/roc_pr_mse_AutoEncoder.svg" alt="ROC and PR curves", width="100%"></td>
    <!-- <td><img src="lab4/plots/people/roc_pr_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/people/roc_pr_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
</table>

<table>
  <caption>Performance on FakeData</caption>
  <tr>
    <!-- <td><img src="lab4/plots/noise/scores_max_logit_LargeCNNskip.svg" alt="Scores from CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/noise/scores_max_softmax_LargeCNNskip.svg" alt="Scores from CNN using max_softmax", width="100%"></td>
    <td><img src="lab4/plots/noise/scores_mse_AutoEncoder.svg" alt="Scores from CNN using max_logit", width="100%"></td>
    <!-- <td><img src="lab4/plots/noise/scores_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/noise/scores_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
  <tr>
    <!-- <td><img src="lab4/plots/noise/roc_pr_max_logit_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td> -->
    <td><img src="lab4/plots/noise/roc_pr_max_softmax_LargeCNNskip.svg" alt="ROC and PR curves", width="100%"></td>
    <td><img src="lab4/plots/noise/roc_pr_mse_AutoEncoder.svg" alt="ROC and PR curves", width="100%"></td>
    <!-- <td><img src="lab4/plots/noise/roc_pr_max_logit_RLargeCNNskip.svg" alt="Scores from robust CNN using max_logit", width="100%"></td> -->
    <td><img src="lab4/plots/noise/roc_pr_max_softmax_RLargeCNNskip.svg" alt="Scores from robust CNN using max_softmax", width="100%"></td>
  </tr>
</table>

On the `lab4/plots/` folder you can find also the plots with the `max_logit` score that are not displayed here, since the `max_softmax` performs slightly better.

</details>
