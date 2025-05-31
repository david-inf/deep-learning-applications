
echo "DistilBERT full-finetuning"
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_full.yaml

echo "DistilBERT finetuning using LoRA on q"
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_q8.yaml
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_q16.yaml

echo "DistilBERT finetuning using LoRA on q,v"
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_qv8.yaml
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_qv16.yaml

echo "DistilBERT finetuning using LoRA on q,v,k,o"
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_qkvo8.yaml
CUDA_VISIBLE_DEVICES="0" accelerate launch lab3/main_ft.py --config lab3/configs/distilbert_lora_qkvo16.yaml