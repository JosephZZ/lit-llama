
python evaluate/aligner_rougeL.py --aligner_length 1 --N -1 --test_role "Doctor Who"  --test_file_path "data/rolebench/rolegpt_baseline.jsonl" --aligner_path "out/aligner/lit-llama-2-eng_instruction_generalization_doctorwho_100/7B/1vector-start_layer2-lr0.005bs32weightDecay0.02wu30/final.pth"

python finetune/lora.py --data_name "rolebench/eng_instruction_generalization_doctorwho_100"

python evaluate/lora_rougeL.py --test_role "Doctor Who" --N -1  --test_file_path "data/rolebench/rolegpt_baseline.jsonl" --lora_path "out/lora/lit-llama-2-rolebench/eng_instruction_generalization_doctorwho_100/7B/lora_r8_alpha16_dropout0.05_lr0.0001_bs32_epoch200_warmup5/lit-llama-lora-finetuned.pth"


# python finetune/aligner.py --aligner_length 10 --data_dir "data/rolebench/eng_instruction_generalization_gaston_100"
# python finetune/aligner.py --aligner_length 1 --data_dir "data/rolebench/eng_instruction_generalization_gaston_100"



# python evaluate/aligner_rougeL.py --aligner_length 1 --N -1 --test_role "Gaston"  --test_file_path "data/rolebench/rolegpt_baseline.jsonl" --aligner_path "out/aligner/lit-llama-2-eng_instruction_generalization_gaston_100/7B/1vector-start_layer2-lr0.009bs32weightDecay0.02wu15/final.pth"
# python evaluate/aligner_rougeL.py --aligner_length 10 --N -1 --test_role "Gaston"  --test_file_path "data/rolebench/rolegpt_baseline.jsonl" --aligner_path "out/aligner/lit-llama-2-eng_instruction_generalization_gaston_100/7B/10vector-start_layer2-lr0.009bs32weightDecay0.02wu15/final.pth"

# python evaluate/lora_rougeL.py --test_role "Gaston" --N -1  --test_file_path "data/rolebench/rolegpt_baseline.jsonl" --lora_path "out/lora/lit-llama-2-rolebench/eng_instruction_generalization_gaston_100/7B/lora_r8_alpha16_dropout0.05_lr0.0001_bs32_epoch200_warmup5/lit-llama-lora-finetuned.pth"
