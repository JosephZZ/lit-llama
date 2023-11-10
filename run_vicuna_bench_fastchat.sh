# CUDA_VISIBLE_DEVICES=0 \
# python generate/aligner_fastchat_bench_jsonl_gen.py \
# --aligner_path  out/aligner/lit-llama-2-alpaca1k256/13B/1vector-start_layer2-lr0.009bs32weightDecay0.02wu10/epoch-40.0-iter-009999.pth \
# --pretrained_path checkpoints/lit-llama-2/13B/lit-llama.pth \
# --aligner_length 1 \
# --aligner_start_layer 2 \
# --num_choices 1


CUDA_VISIBLE_DEVICES=1 \
python generate/lora_fastchat_bench_jsonl_gen.py \
--lora_path out/lora/lit-llama-2-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs64_epoch5/epoch5-iter-032479-ckpt.pth \
--pretrained_path checkpoints/lit-llama-2/7B/lit-llama.pth  \
--lora_r 8 