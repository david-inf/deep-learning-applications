MODEL=lab1/configs/CNN/LargeCNNskip.yaml
# MODEL=lab4/ckpts/cnn_robust.yaml
# MODEL=lab1/configs/WideResNet/WideResNet14-2.yaml

SEED=42

python lab4/main_adversarial.py --attack untargeted --model_configs $MODEL --seed $SEED --n_samples 7

python lab4/main_adversarial.py --attack targeted --model_configs $MODEL --target deer --seed $SEED --n_samples 7