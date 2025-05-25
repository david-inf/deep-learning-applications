echo "OOD dataset: FakeData"

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood noise
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood noise
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs lab4/ckpts/autoencoder.yaml --ood noise

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood people
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood people
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs lab4/ckpts/autoencoder.yaml --ood people

echo "CNN with max_logit"
python lab4/main_detection.py --score_fun max_logit --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood aquatic
echo "CNN with max_softmax"
python lab4/main_detection.py --score_fun max_softmax --model_configs lab1/configs/CNN/LargeCNNskip.yaml --ood aquatic
echo "AutoEncoder"
python lab4/main_detection.py --score_fun mse --model_configs lab4/ckpts/autoencoder.yaml --ood aquatic