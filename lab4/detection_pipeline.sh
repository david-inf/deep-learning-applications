FILE=lab4/main_detection.py
CNN=lab1/configs/CNN/LargeCNNskip.yaml
AE=lab4/ckpts/autoencoder.yaml
RCNN=lab4/ckpts/cnn_robust.yaml

echo "OOD dataset: FakeData"

echo "CNN with max_logit"
python $FILE --score_fun max_logit --model_configs $CNN --ood noise
echo "CNN with max_softmax"
python $FILE --score_fun max_softmax --model_configs $CNN --ood noise
echo "AutoEncoder"
python $FILE --score_fun mse --model_configs $AE --ood noise
echo "Robust model"
python $FILE --score_fun max_logit --model_configs $RCNN --ood noise

echo "OOD dataset: people subset"

echo "CNN with max_logit"
python $FILE --score_fun max_logit --model_configs $CNN --ood people
echo "CNN with max_softmax"
python $FILE --score_fun max_softmax --model_configs $CNN --ood people
echo "AutoEncoder"
python $FILE --score_fun mse --model_configs $AE --ood people
echo "Robust model"
python $FILE --score_fun max_logit --model_configs $RCNN --ood people

echo "OOD dataset: aquatic mammals subset"

echo "CNN with max_logit"
python $FILE --score_fun max_logit --model_configs $CNN --ood aquatic
echo "CNN with max_softmax"
python $FILE --score_fun max_softmax --model_configs $CNN --ood aquatic
echo "AutoEncoder"
python $FILE --score_fun mse --model_configs $AE --ood aquatic
echo "Robust model"
python $FILE --score_fun max_logit --model_configs $RCNN --ood aquatic