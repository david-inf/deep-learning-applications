MODEL=lab4/ckpts/cnn_robust.yaml

echo "Adversarial training"
python lab4/main_robust.py --config $MODEL

echo "OOD detection pipeline on FakeData using max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs $MODEL --ood noise