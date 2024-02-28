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


IGNORE_INDEX = -1



def prepare(
    destination_path: Path = Path("data/baize"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_ratio: float = 0.1,  # default 90% train, 10% validation
    max_seq_length: int = 1536,
    seed: int = 42,
) -> None:
    """Prepare any dataset for finetuning (akin to Shakespheare full tuning).

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)
    # file_path = destination_path / data_file_name
    # if not file_path.exists():
    #     raise AssertionError(f"{data_file_name} is provided by the user")

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)

    data = []

    # Loop through every file in the directory
    for filename in os.listdir(destination_path):
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            # Create the full file path by joining the directory and filename
            filepath = os.path.join(destination_path, filename)
            
            # Open the JSON file and print its content
            with open(filepath, 'r') as input_file:
                this_data = json.load(input_file)
                print(f"Loaded {len(this_data)} samples from {filename}")
                data += this_data

    item_length = []
    for item in data:
        item.pop("topic")
        item_length.append(len(item["input"]))

    print(f"Loaded a total of {len(data)} samples")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(item_length, 25))
    print("50th percentile: ", np.percentile(item_length, 50))
    print("75th percentile: ", np.percentile(item_length, 75))
    print("85th percentile: ", np.percentile(item_length, 85))
    print("95th percentile: ", np.percentile(item_length, 95))
    print("99th percentile: ", np.percentile(item_length, 99))
    print("max one: ", max(item_length))
    # Partition the dataset into train and test
    test_split_size = min(int(len(data) * test_split_ratio),2000)

    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data,
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    encoded_full_prompt = tokenize(tokenizer, line['input'], max_length=max_length, eos=False)
    return {
        "input_ids": encoded_full_prompt,
        "labels": encoded_full_prompt,
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)