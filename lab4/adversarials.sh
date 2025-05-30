MODEL=lab1/configs/CNN/LargeCNNskip.yaml
SEED=111

python lab4/main_adversarial.py --attack untargeted --model_configs $MODEL --seed $SEED

python lab4/main_adversarial.py --attack targeted --model_configs $MODEL --target deer --seed $SEED