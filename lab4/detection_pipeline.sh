CNN=lab1/configs/CNN/LargeCNNskip.yaml
AE=lab4/ckpts/autoencoder.yaml

echo "OOD dataset: FakeData"

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs $CNN --ood noise
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs $CNN --ood noise
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs $AE --ood noise

echo "OOD dataset: people subset"

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs $CNN --ood people
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs $CNN --ood people
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs $AE --ood people

echo "OOD dataset: aquatic mammals subset"

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs $CNN --ood aquatic
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs $CNN --ood aquatic
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs $AE --ood aquatic