echo "MNIST"
python lab1/main_train.py --config lab1/configs/MLP/MLP_mnist.yaml
python lab1/main_train.py --config lab1/configs/CNN/CNN1_mnist.yaml
python lab1/main_train.py --config lab1/configs/CNN/CNN2_mnist.yaml

echo "Train CNNs"
python lab1/main_train.py --config lab1/configs/CNN/SmallCNN.yaml
python lab1/main_train.py --config lab1/configs/CNN/MediumCNN.yaml
python lab1/main_train.py --config lab1/configs/CNN/LargeCNN.yaml

echo "Train ResNet"
python lab1/main_train.py --config lab1/configs/ResNet/ResNet32.yaml
python lab1/main_train.py --config lab1/configs/ResNet/ResNet44.yaml
python lab1/main_train.py --config lab1/configs/ResNet/ResNet56.yaml

echo "Knowledge Distillation"
python lab1/main_distil.py --config lab1/configs/Distil/DistilCNN_RN56.yaml
