"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path
import os
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

import datasets

TRAIN_DATA_FILE_NAME = "train.json"
TEST_DATA_FILE_NAME = "test.json"
IGNORE_INDEX = -1

def prepare(
    data_folder: Path = Path("data/hh"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    test_split_ratio: float = 0.1,
    max_sys_prompt_length: int = 128,
    max_seq_length: int = 1024,
    seed: int = 42,
    is_for_debug = False, 
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Dolly dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    

    # destination_path.mkdir(parents=True, exist_ok=True)
    data_folder = data_folder
    train_file_path = data_folder / TRAIN_DATA_FILE_NAME
    test_file_path =  data_folder / TEST_DATA_FILE_NAME

    if not os.path.exists(train_file_path) \
        or not os.path.exists(test_file_path):
        ds = datasets.load_dataset("Anthropic/hh-rlhf")
        with open(train_file_path, 'w', encoding='utf-8') as f:
            dataset_list = [dict(example) for example in ds['train']]
            json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            dataset_list = [dict(example) for example in ds['test']]
            json.dump(dataset_list, f, ensure_ascii=False, indent=4)

    # # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)

    with open(train_file_path, "r",encoding='utf-8') as file:
        # data_train = file.readlines()
        # data_train = [json.loads(line) for line in data_train]
        data_train = json.load(file)

        # data = json.load(file)
    
    with open(test_file_path, "r", encoding='utf-8') as file:
        # data_test = file.readlines()
        # data_test = [json.loads(line) for line in data_test]
        data_test = json.load(file)



    chosen_resp_length = []
    rejected_resp_length = []
    for item in data_train:
        chosen_resp_length.append(len(item["chosen"]))
        rejected_resp_length.append(len(item["rejected"]))

    print(f"Loaded a total of {len(data_train)} samples")
    print("chosen")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(chosen_resp_length, 25))
    print("50th percentile: ", np.percentile(chosen_resp_length, 50))
    print("75th percentile: ", np.percentile(chosen_resp_length, 75))
    print("85th percentile: ", np.percentile(chosen_resp_length, 85))
    print("95th percentile: ", np.percentile(chosen_resp_length, 95))
    print("99th percentile: ", np.percentile(chosen_resp_length, 99))
    print("max one: ", max(chosen_resp_length))


    print("rejected")
    print("25th percentile: ", np.percentile(rejected_resp_length, 25))
    print("50th percentile: ", np.percentile(rejected_resp_length, 50))
    print("75th percentile: ", np.percentile(rejected_resp_length, 75))
    print("85th percentile: ", np.percentile(rejected_resp_length, 85))
    print("95th percentile: ", np.percentile(rejected_resp_length, 95))
    print("99th percentile: ", np.percentile(rejected_resp_length, 99))
    print("max one: ", max(rejected_resp_length))


    # Partition the dataset into train and test

    train_set, test_set = list(data_train), list(data_test)
    if is_for_debug:
        train_set = train_set[:10]
        test_set = test_set[:10]

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, data_folder / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, data_folder / "test.pt")


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    chosen_resp = tokenize(tokenizer, line['chosen'], max_length=max_length, eos=False)
    rejected_resp = tokenize(tokenizer, line['rejected'], max_length=max_length, eos=False)
    return {
        "chosen": chosen_resp,
        "rejected": rejected_resp,
        "label_chosen" : chosen_resp,
        "label_rejected" : rejected_resp
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
