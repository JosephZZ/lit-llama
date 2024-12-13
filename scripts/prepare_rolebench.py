"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

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

DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
TRAIN_DATA_FILE_NAME = "train.jsonl"
TEST_DATA_FILE_NAME = "test.jsonl"
ROLE = "Sheldon Cooper"
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/rolebench"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    test_split_size: int = 2000,
    number_of_train: int = 100, #None for all
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    train_data_file_name: str = TRAIN_DATA_FILE_NAME,
    test_data_file_name = TEST_DATA_FILE_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    train_file_path = destination_path / train_data_file_name
    test_file_path = destination_path / test_data_file_name

    # download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    with open(train_file_path, "r") as file:
        # data = json.load(file)
        data_train = file.readlines()
        data_train = [json.loads(line) for line in data_train]

    with open(test_file_path, "r") as file:
        # data = json.load(file)
        data_test = file.readlines()
        data_test = [json.loads(line) for line in data_test]

    def flatten_data_sample(data, role, N=None):
        all_flatten = []
        for sample in data:
            for i in range(len(sample["generated"])):
                if sample["role"] != role:
                    continue
                if N is not None and len(all_flatten) >= N:
                    break
                flatten_sample = {}
                flatten_sample["role"] = sample["role"]
                flatten_sample["question"] = sample["question"]
                flatten_sample["generated"] = sample["generated"][i]
                all_flatten.append(flatten_sample)
        return all_flatten

    data_train = flatten_data_sample(data_train, ROLE, number_of_train) 
    data_test = flatten_data_sample(data_test, ROLE)

    sample_full_length = [count_sample_length(sample) for sample in data_train]
    print(f"Loaded a total of {len(sample_full_length)} samples")
    # Calculate and print the percentiles
    print("25th percentile: ", np.percentile(sample_full_length, 25))
    print("50th percentile: ", np.percentile(sample_full_length, 50))
    print("75th percentile: ", np.percentile(sample_full_length, 75))
    print("85th percentile: ", np.percentile(sample_full_length, 85))
    print("95th percentile: ", np.percentile(sample_full_length, 95))
    print("99th percentile: ", np.percentile(sample_full_length, 99))
    print("max one: ", max(sample_full_length))

    # Partition the dataset into train and test
    # train_split_size = len(data) - test_split_size
    # train_set, test_set = random_split(
    #     data, 
    #     lengths=(train_split_size, test_split_size),
    #     generator=torch.Generator().manual_seed(seed),
    # )
    train_set, test_set = list(data_train), list(data_test)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, train_file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, test_file_path.parent / "test.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)

def count_sample_length(example):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["generated"]
    return len(full_prompt_and_response)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The input text is formed as a single message including all
    the instruction, the input (optional) and the response.
    The label/target is the same message but can optionally have the instruction + input text
    masked out (mask_inputs=True).

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["generated"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['question']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
