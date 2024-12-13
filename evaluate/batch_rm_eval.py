import torch
import os
import json
import re
import ipdb
import numpy as np 
# from config import *
from typing import Dict, List
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
# from safe_rlhf.models import AutoModelForScore

import pandas as pd
def load_json(fp):

    with open (os.path.join(fp), 'r') as f:
        datas = json.load(f)

    return datas

def load_json_line(fp):
    with open (os.path.join(fp), 'r') as f:
        datas = [json.loads(line) for line in f]

    return datas

def plot_fig(x, y):
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制折线图
    ax.plot(x, y, marker='o', linestyle='-', color='b', label='Data Line')

    # 添加标题和标签
    ax.set_title('avg reward relation')
    ax.set_xlabel('token id')
    ax.set_ylabel('reward')

    # 添加网格
    # ax.grid(True)
    # 添加图例
    ax.legend()
    # 显示图形
    plt.show()


def batch_armorm_score(rmodel, tokenizer, max_length, device, messages: List[List[Dict[str, str]]]) -> List[float]:
    batch_input = []
    for message in messages:
        message_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        batch_input.append(message_text)
    
    batch_inputs = tokenizer(batch_input, padding=True, truncation=True, add_special_tokens=True, 
                                  max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = batch_inputs.input_ids.to(device), batch_inputs.attention_mask.to(device)
        
    with torch.no_grad():
        output = rmodel(input_ids=input_ids, attention_mask=attention_mask)
        # Multi-objective rewards for the response
        multi_obj_rewards = output.rewards.cpu().float()
    
    return multi_obj_rewards

def batch_mistral_output(tokenizer, pipeline, messages: List[List[Dict[str, str]]]):

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }

    input = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")]
    pipe_outputs = pipeline(input, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards

def batch_beavertails_score(rmodel, tokenizer, max_length, device, messages: List[List[Dict[str, str]]]) -> List[float]:
    batch_input = []
    formatted = "BEGINNING OF CONVERSATION: "
    for msg in messages:
        formatted += f"{msg['role'].upper()}: {msg['content']} "
    formatted = formatted.strip()
    inputs = tokenizer(formatted, return_tensors='pt')
    input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)        
    with torch.no_grad():
        output = rmodel(input_ids=input_ids, attention_mask=attention_mask)

    return output.logits.cpu().float()

def get_alpaca_qas(p_path, a_path):
    prompts = load_json_line(os.path.join(f"{p_path}"))
    responses = load_json_line(os.path.join(f"{a_path}"))
    datas = []
    for prompt, response in zip(prompts, responses):
        datas.append({"prompt": prompt['turns'][0], "response": response['choices'][0]['turns'][0]})

    return datas

def get_beaver_qas(path):
    json_data = load_json(os.path.join(f"{path}"))
    datas = []
    for data in json_data:
        datas.append({"prompt": data['instruction'], "response": data['response']})

    return datas

def load_reward_model(model_path: str, device: str):
    """Load the reward model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        device_map=device,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer

def evaluate_mistral_responses(
    model_path: str,
    eval_config: dict,
    max_length: int = 128,
    device: str = None,
    attributes: List[str] = None
) -> dict:
    """Main evaluation function that processes responses and returns rewards."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    rm_pipe = pipeline(
      "sentiment-analysis",
      model="checkpoints/RM-Mistral-7B",
      #device="auto",
      device=device,
      tokenizer=tokenizer,
      model_kwargs={"torch_dtype": torch.bfloat16}
    )

    # Get evaluation data
    path_dir = eval_config['path_dir']
    cwd = os.getcwd()
    rwd = {}
    
    for token_idx, file in enumerate(os.listdir(path_dir)):
        rwd[file] = []
        
        # Load QA pairs based on data type
        if 'beaver' in path_dir:
            datas = get_beaver_qas(os.path.join(cwd, f"{path_dir}/{file}"))
        elif 'alpaca' in path_dir:
            datas = get_alpaca_qas(
                eval_config['alpaca_prompt_path'],
                os.path.join(cwd, f"{path_dir}/{file}")
            )
        
        # Process each QA pair
        for data in datas:
            messages = [
                {"role": "user", "content": data['prompt']},
                {"role": "assistant", "content": data['response']}
            ]
            
            rwd[file] += batch_mistral_output(tokenizer, rm_pipe, messages)
        
        print(f'The average reward for {file} is {np.array(rwd[file]).mean()}.')
    
    return rwd

def evaluate_beavertails_responses(
    model_path: str,
    eval_config: dict,
    max_length: int = 128,
    device: str = None,
    attributes: List[str] = None
) -> dict:
    """Main evaluation function that processes responses and returns rewards."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_reward_model(model_path, device)

    # Get evaluation data
    path_dir = eval_config['path_dir']
    cwd = os.getcwd()
    rwd = {}
    
    for token_idx, file in enumerate(os.listdir(path_dir)):
        rwd[file] = []
        
        # Load QA pairs based on data type
        if 'beaver' in path_dir:
            datas = get_beaver_qas(os.path.join(cwd, f"{path_dir}/{file}"))
        elif 'alpaca' in path_dir:
            datas = get_alpaca_qas(
                eval_config['alpaca_prompt_path'],
                os.path.join(cwd, f"{path_dir}/{file}")
            )
        
        # Process each QA pair
        for data in datas:
            messages = [
                {"role": "user", "content": data['prompt']},
                {"role": "assistant", "content": data['response']}
            ]
            
            rwd[file] += batch_beavertails_score(model, tokenizer, max_length, device, messages)
        rwd[file] = torch.stack(rwd[file])
        avg_rwd = rwd[file].mean(dim=0)
        print(f'The average reward for {file} is {avg_rwd}.')
    
    return rwd


def evaluate_armorm_responses(
    model_path: str,
    eval_config: dict,
    max_length: int = 128,
    device: str = None,
    attributes: List[str] = None
) -> dict:
    """Main evaluation function that processes responses and returns rewards."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_reward_model(model_path, device)

    # Get evaluation data
    path_dir = eval_config['path_dir']
    cwd = os.getcwd()
    rwd = {}

    # create a dataframe to store the rewards, the columns are the attributes of the reward model, the rows are the files in the path_dir 
    df = pd.DataFrame(columns=['file_name', 'qa_pair_id', 'prompt', 'response']+attributes)
    summary_df_1 = pd.DataFrame(columns=['file_name']+attributes)
    summary_df_2 = pd.DataFrame(columns=['file_name']+attributes)

    filtered_summary_df_1 = pd.DataFrame(columns=['file_name']+attributes)
    filtered_summary_df_2 = pd.DataFrame(columns=['file_name']+attributes)
    for token_idx, file in enumerate(os.listdir(path_dir)):
        file_name = '_'.join(file.split('_')[0:2])
        
        rwd[file_name] = []

        # Load QA pairs based on data type
        if 'beaver' in path_dir:
            datas = get_beaver_qas(os.path.join(cwd, f"{path_dir}/{file}"))
        elif 'alpaca' in path_dir:
            datas = get_alpaca_qas(
                eval_config['alpaca_prompt_path'],
                os.path.join(cwd, f"{path_dir}/{file}")
            )
    
        # Process each QA pair
        for i, data in enumerate(datas):
            messages = [
                {"role": "user", "content": data['prompt']},
                {"role": "assistant", "content": data['response']}
            ]

            method_rwd = batch_armorm_score(model, tokenizer, max_length, device, [messages]).mean(dim=0)
            # Get reward for specified attribute
            rwd[file_name].append(method_rwd)

            new_row = dict(zip(attributes, method_rwd.numpy()))
            df.loc[len(df)] = [file_name, i, data['prompt'], data['response']] + list(new_row.values())

        rwd[file_name] = torch.stack(rwd[file_name])
        print(f'The average reward for {file_name} is {rwd[file_name].mean(dim=0)}.')
        summary_df_1.loc[len(summary_df_1)] = [file_name] + list(rwd[file_name].mean(dim=0).numpy())
        summary_df_2.loc[len(summary_df_2)] = [file_name] + list(rwd[file_name].std(dim=0).numpy())

        # for each of the attribute, remove the outliers 2 std away and get the mean and std of the remaining data
        filtered_rwd_mean = []
        filtered_rwd_std = []
        for attr in attributes:
            attr_data = rwd[file_name][:, attributes.index(attr)]
            mean = attr_data.mean()
            std = attr_data.std()
            threshold = 2 * std
            filtered_attr_data = attr_data[(attr_data >= mean - threshold) & (attr_data <= mean + threshold)]
            filtered_rwd_mean.append(filtered_attr_data.mean().numpy())
            filtered_rwd_std.append(filtered_attr_data.std().numpy())
        filtered_summary_df_1.loc[len(filtered_summary_df_1)] = [file_name] + filtered_rwd_mean
        filtered_summary_df_2.loc[len(filtered_summary_df_2)] = [file_name] + filtered_rwd_std

    #save the dataframe to a csv file
    prefix = eval_config['path_dir'].split('/')[-1]
    df.to_csv(f'{prefix}_armorm_eval_result.csv', index=False)
    summary_df_1.to_csv(f'{prefix}_armorm_summary_mean.csv', index=False)
    summary_df_2.to_csv(f'{prefix}_armorm_summary_std.csv', index=False)

    filtered_summary_df_1.to_csv(f'{prefix}_armorm_filtered_summary_mean.csv', index=False)
    filtered_summary_df_2.to_csv(f'{prefix}_armorm_filtered_summary_std.csv', index=False)

    # for each of the attribute, get the file name that has the highest reward
    for attr in attributes:
        max_rwd_file = summary_df_1[attr].idxmax()
        max_file_name = summary_df_1.loc[max_rwd_file]['file_name']
        
        filtered_max_rwd_file = filtered_summary_df_1[attr].idxmax()
        filtered_max_file_name = filtered_summary_df_1.loc[filtered_max_rwd_file]['file_name']
        print(f'The file name that has the highest reward for {attr} is {max_file_name}.')
        print(f'The file name that has the highest reward for {attr} after filtering outliers is {filtered_max_file_name}.')

def main():
    # The attributes of the 19 reward objectives
    armorm_attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']


    # Example configuration
    eval_config = {
        'path_dir': "out/beaver_responses",
        # 'path_dir': "out/alpaca_eval",
        'alpaca_prompt_path': "data/evaluation/Vicuna_questions.jsonl",
    }
      
    # Run evaluation
    armorm_model_path = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
    armorm_rewards = evaluate_armorm_responses(
        model_path=armorm_model_path,
        eval_config=eval_config,
        attributes=armorm_attributes
    )

    # beavertails_model_path = 'PKU-Alignment/beaver-7b-v1.0-reward'
    # beavertails_rewards = evaluate_beavertails_responses(
    #     model_path=beavertails_model_path,
    #     eval_config=eval_config,
    # )
    
    # mistral_rw_path = 'checkpoints/RM-Mistral-7B'
    # mistral_rewards = evaluate_mistral_responses(
    #     model_path=mistral_rw_path,
    #     eval_config=eval_config,
    # )


    # Optionally plot results
    # if plot_results:
    #     plot_fig([i for i in range(1, max_length + 1)], rewards)

if __name__ == "__main__":
    main()


