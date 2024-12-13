# CUDA_VISIBLE_DEVICES=0 \
# python generate/promptAligner.py \
# --aligner_path  out/promptAlignerV2/lit-llama-2-philosopherQA_mydata/7b-chat/1vector-start_layer2-lr2e-05bs16/epoch-20.0.pth \
# --pretrained_path checkpoints/lit-llama-2/7b-chat/lit-llama.pth \
# --tokenizer_path checkpoints/lit-llama-2/tokenizer.model \
# --max_new_tokens 200 \
# --temperature 0.7 \
# --system_align_prompt "Michel Foucault (1926-1984 CE): A French philosopher, known for his critical studies of social institutions, including mental institutions, prisons, and the sciences.\nYou are an assistant taking on the perspective and value of this person." \
# --prompt "What is the relationship between morality and law?"

# CUDA_VISIBLE_DEVICES=0 \
# python generate/promptAlignerV4.py \
# --aligner_path  out/promptAlignerV2/lit-llama-2-orca/7b-chat/1vector-start_layer2-lr0.0009bs8/epoch-1.0.pth \
# --pretrained_path checkpoints/lit-llama-2/7b-chat/lit-llama.pth \
# --tokenizer_path checkpoints/lit-llama-2/tokenizer.model \
# --max_new_tokens 200 \
# --temperature 0.7 \
# --system_align_prompt "You are an AI assistant. You will be given a task. You must generate a detailed and long answer." \
# --prompt "Answer the following question:\n\nhow many stations does the jerusalem light rail have?" \

CUDA_VISIBLE_DEVICES=0 \
python generate/promptAlignerV8NoBase.py \
--aligner_path  out/promptAlignerV8NoBase/lit-llama-2-orca/7B/1vector-start_layer2-lr1e-06bs32/epoch-3.0.pth \
--pretrained_path checkpoints/lit-llama-2/7B/lit-llama.pth \
--tokenizer_path checkpoints/lit-llama-2/tokenizer.model \
--max_new_tokens 200 \
--temperature 0.7 \
--system_align_prompt "You are an AI assistant. You will be given a task. You must generate a detailed and long answer." \
--prompt "Answer the following question:\n\nhow many stations does the jerusalem light rail have?" \
