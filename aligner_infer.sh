## use 1 vector
CUDA_VISIBLE_DEVICES=0 \
python generate/aligner.py \
--aligner_path out/aligner/llama-alpaca/13B/1vector/lit-llama_alpaca-aligner-1vectors-finetuned.pth \
--pretrained_path checkpoints/lit-llama/13B/lit-llama.pth \
--tokenizer_path checkpoints/lit-llama/tokenizer.model \
--max_new_tokens 300 \
--temperature 0.7 \
--prompt "As a superhero, how would you explain your origin story to a curious child?"
## use 10 vectors
#python generate/aligner.py \
#--aligner_path /home/ziheng/sharedWithLocal/llm/lit-llama/out/aligner/alpaca/openllama-7B/10vectors/lit-openllama_alpaca-aligner-10vectors-finetuned.pth \
#--pretrained_path /home/ziheng/sharedWithLocal/llm/lit-llama/checkpoints/lit-open-llama/7B/lit-open-llama.pth \
#--tokenizer_path /home/ziheng/sharedWithLocal/llm/lit-llama/checkpoints/lit-open-llama/tokenizer.model \
#--max_new_tokens 500 \
#--temperature 0.8 \
#--prompt "Write a Python program that prints the first 10 Fibonacci numbers."
# # use 100 vectors
# python generate/aligner.py \
# --aligner_path /home/ziheng/sharedWithLocal/llm/lit-llama/out/aligner/alpaca/openllama-7B/100vectors/lit-openllama_alpaca-aligner-100vectors-finetuned.pth \
# --pretrained_path /home/ziheng/sharedWithLocal/llm/lit-llama/checkpoints/lit-open-llama/7B/lit-open-llama.pth \
# --tokenizer_path /home/ziheng/sharedWithLocal/llm/lit-llama/checkpoints/lit-open-llama/tokenizer.model \
# --max_new_tokens 500 \
# --temperature 0.8 \
# --prompt "How can i develop my critical thinking skills? Give me three suggestions."
