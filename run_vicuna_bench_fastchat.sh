CUDA_VISIBLE_DEVICES=3\
python generate/aligner_fastchat_bench_jsonl_gen.py \
        --aligner_path out/aligner/llama-dolly/7B/aligner-1vector-start_layer2-lr9e-05bs63/epoch-5.03994-iter-083999.pth \
        --pretrained_path checkpoints/lit-llama/7B/lit-llama.pth \
        --aligner_length 1 \
        --aligner_start_layer 2 \

