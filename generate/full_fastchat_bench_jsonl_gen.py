import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm
import os 
import shortuuid

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.model import LLaMA
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
# from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.7,
    question_file: Path = Path("data/evaluation/Vicuna_questions.jsonl"),
    num_choices: int = 1,
    file_suffix : str = "",
    is_multi_turn: bool = False,
    instruct_style = "alpaca_3_shot"
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        aligner_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()
    answer_file = str(pretrained_path)[:-3] + question_file.name[:-5] + "_answers" + file_suffix + ".jsonl"
    print("path: ", answer_file)
    
    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)


    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)

    if question_file is None:
        sample = {"instruction": prompt, "input": input}
        prompt = generate_prompt(sample,instruct_style=instruct_style)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        print(output)

        tokens_generated = y.size(0) - prompt_length
        print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    else:
        questions = load_questions(question_file, 0, 100)
        for question in tqdm(questions):
            # if question["category"] in temperature_config:
            #     temperature = temperature_config[question["category"]]
            # else:
            #     temperature = 0.7

            choices = []
            for i in range(num_choices):
                torch.manual_seed(i)
                turns = []
                history = []
                for j in range(len(question["turns"])):
                    sample = {"instruction": question["turns"][j], "input": ""}
                    prompt = generate_prompt(sample,instruct_style=instruct_style)

                    print("\n\n Turn {} with prompt length {}: \n\n {}".format(j,len(prompt), prompt))
                    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

                    # some models may error out when generating long outputs
                    try:
                        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
                        # print('could run model ', y.size())
                        model.reset_cache()
                        output = tokenizer.decode(y)
                        # print('could decode ', output)
                        last_response = output.split("### Response:")[j+1].strip()
                        print(last_response)

                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"

                    turns.append(last_response)
                    sample["output"]=last_response
                    history.append(sample)
                 
                choices.append({"index": i, "turns": turns})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": pretrained_path.name,  #get the model name
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
    reorg_answer_file(answer_file)

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

# # Sampling temperature configs for
# temperature_config = {
#     "writing": 0.7,
#     "roleplay": 0.7,
#     "extraction": 0.0,
#     "math": 0.0,
#     "coding": 0.0,
#     "reasoning": 0.0,
#     "stem": 0.1,
#     "humanities": 0.1,
# }

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])



def generate_prompt(example, instruct_style):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    if instruct_style == "alpaca":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    elif instruct_style == "alpacaLong":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a high quality and expanded response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    elif instruct_style == "orca" or  instruct_style ==  "baize":
        return "[|User|] "+ example['instruction']+" [|AI Assistant|] "
    elif instruct_style == "beaver":
        return "[User] "+ example['instruction']+" [Assistant] "
    elif instruct_style == "alpaca_zero_shot":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request. The response should consider aspects such as helpfulness, relevance, accuracy, depth, creativity, and level of detail.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    elif instruct_style == "alpaca_one_shot":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request. The response should consider aspects such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. \
            For example, \n \
            Instruction: Give three tips for staying healthy. \
            Response: Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule. \
            \n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    elif instruct_style == "alpaca_3_shot":
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request. The response should consider aspects such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. \
            For examples, \n \
            Instruction: Give three tips for staying healthy. \
            Response: Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule. \
            \n \
            Instruction: What are the three primary colors? \
            Response: The three primary colors are red, blue, and yellow. \
            \n \
            Instruction: Describe the structure of an atom. \
            Response: An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom. \
            \n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )
    else:
        raise ValueError(f"Unknown instruction style: {instruct_style}")
    

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
