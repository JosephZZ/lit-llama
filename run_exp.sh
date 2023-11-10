# CUDA_VISIBLE_DEVICES=0 python finetune/adapter.py
# CUDA_VISIBLE_DEVICES=0 python finetune/adapterDPO.py

# CUDA_VISIBLE_DEVICES=1 python finetune/lora.py
# CUDA_VISIBLE_DEVICES=1 python finetune/loraDPO.py

CUDA_VISIBLE_DEVICES=0,1 python finetune/aligner.py
CUDA_VISIBLE_DEVICES=0,1 python finetune/alignerDPO.py