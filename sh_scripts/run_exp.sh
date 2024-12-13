# CUDA_VISIBLE_DEVICES=0 python finetune/adapter.py
# CUDA_VISIBLE_DEVICES=0 python finetune/adapterDPO.py

# CUDA_VISIBLE_DEVICES=1 python finetune/lora.py
# CUDA_VISIBLE_DEVICES=1 python finetune/loraDPO.py

# CUDA_VISIBLE_DEVICES=0,1 python finetune/aligner.py
# CUDA_VISIBLE_DEVICES=0,1 python finetune/alignerDPO.py

python evaluate/run_safety_gpt_eval.py --result_file alignerBlue_vs_0shotRed --red_corner_model_output_name_or_path /home/zhouziheng/Desktop/lit-llama/out/full/beaver_safety_7B/alpaca_beaversafe_zero_shot_results.json
python evaluate/run_safety_gpt_eval.py --result_file alignerBlue_vs_3shotRed --red_corner_model_output_name_or_path /home/zhouziheng/Desktop/lit-llama/out/full/beaver_safety_7B/alpaca_beaversafe_3_shot_results.json
python evaluate/run_safety_gpt_eval.py --result_file alignerBlue_vs_1shotRed --red_corner_model_output_name_or_path /home/zhouziheng/Desktop/lit-llama/out/full/beaver_safety_7B/alpaca_beaversafe_1_shot_results.json


