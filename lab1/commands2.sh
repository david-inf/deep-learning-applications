echo "Train CNNs with skip connections"
python lab1/main_train.py --config lab1/configs/CNN/SmallCNNskip.yaml
python lab1/main_train.py --config lab1/configs/CNN/MediumCNNskip.yaml
python lab1/main_train.py --config lab1/configs/CNN/LargeCNNskip.yaml

echo "Train CNN that will be the student"
python lab1/main_train.py --config lab1/configs/CNN/BaseCNN.yaml

echo "Train WideResNet"
python lab1/main_train.py --config lab1/configs/WideResNet/WideResNet14-2.yaml
python lab1/main_train.py --config lab1/configs/WideResNet/WideResNet14-4.yaml
python lab1/main_train.py --config lab1/configs/WideResNet/WideResNet26-4.yaml

echo "Knowledge Distillation"
python lab1/main_distil.py --config lab1/configs/Distil/DistilCNN_WRN26-4.yaml
