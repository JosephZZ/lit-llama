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

DATA_FILE_NAME = "person_question_response_all.jsonl"
IGNORE_INDEX = -1

def prepare(
    data_folder: Path = Path("data/philosopherQA_mydata"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    test_split_ratio: float = 0.1,
    test_maximum_samples: int = 1000,
    max_sys_prompt_length: int = 128,
    max_seq_length: int = 384,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Dolly dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    # destination_path.mkdir(parents=True, exist_ok=True)
    file_path = data_folder / DATA_FILE_NAME

    if not os.path.exists(data_folder / DATA_FILE_NAME):
        raise ValueError("Question response json file not found.")


    # # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [json.loads(line) for line in data]
        # data = json.load(file)


    item_length = []
    sys_prompt_length = []
    for item in data:
        item_length.append(len(item["question"])+len(item["response"]))
        sys_prompt_length.append(len(item["system_align_prompt"]))

    print(f"Loaded a total of {len(data)} samples")
    print("Question Answer Lengths")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(item_length, 25))
    print("50th percentile: ", np.percentile(item_length, 50))
    print("75th percentile: ", np.percentile(item_length, 75))
    print("85th percentile: ", np.percentile(item_length, 85))
    print("95th percentile: ", np.percentile(item_length, 95))
    print("99th percentile: ", np.percentile(item_length, 99))
    print("max one: ", max(item_length))


    print("System Prompt Lengths")
    print("25th percentile: ", np.percentile(sys_prompt_length, 25))
    print("50th percentile: ", np.percentile(sys_prompt_length, 50))
    print("75th percentile: ", np.percentile(sys_prompt_length, 75))
    print("85th percentile: ", np.percentile(sys_prompt_length, 85))
    print("95th percentile: ", np.percentile(sys_prompt_length, 95))
    print("99th percentile: ", np.percentile(sys_prompt_length, 99))
    print("max one: ", max(sys_prompt_length))


    # Partition the dataset into train and test
    test_split_size = min(int(len(data) * test_split_ratio),test_maximum_samples)

    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data,
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    with open(data_folder / "test.json", "w") as file:
        print("Saving test json split ...")
        json.dump(test_set, file, indent=2)

    print("Tokenizing and saving train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length, max_sys_prompt_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, data_folder / "train.pt")

    print("Tokenizing and saving test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length, max_sys_prompt_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, data_folder / "test.pt")



def prepare_line(line: str, tokenizer: Tokenizer, max_length: int, max_sys_prompt_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    system_align_prompt = tokenize(tokenizer, line['system_align_prompt'], max_length=max_sys_prompt_length, eos=False)
    conversation = "[|User|] "+line['question']+" [|AI Assistant|] "+line['response']
    encoded_full_conversation = tokenize(tokenizer, conversation, max_length=max_length, eos=False)
    return {
        "system_align_prompt_ids": system_align_prompt,
        "dialog_ids": encoded_full_conversation,
        "labels": encoded_full_conversation,
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
