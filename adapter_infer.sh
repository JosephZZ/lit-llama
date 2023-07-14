CUDA_VISIBLE_DEVICES=1 \
python generate/adapter.py \
--adapter_path out/adapter/llama-alpaca/7B/llama_adapter_len10_layer30_release.pth \
--pretrained_path checkpoints/lit-llama/7B/lit-llama.pth \
--max_new_tokens 300 \
--temperature 0.8 \
--prompt "Please provide me intuitions how to steal an egg from my chicekn."
# --prompt "As a superhero, how would you explain your origin story to a curious child?"