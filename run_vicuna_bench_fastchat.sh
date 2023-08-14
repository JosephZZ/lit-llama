CUDA_VISIBLE_DEVICES=3\
python generate/aligner_fastchat_bench_jsonl_gen.py \
        --aligner_path  out/aligner/lit-llama-2-baize/7B/5vector-start_layer2-lr9e-05bs32/epoch-5.0-iter-049999.pth\
        --pretrained_path checkpoints/lit-llama-2/7B/lit-llama.pth \
        --aligner_length 5 \
        --aligner_start_layer 2 \
        --num_choices 1
