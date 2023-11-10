import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import pickle
from tqdm import tqdm
import json
from utils import is_equiv
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.modelMultiForward import LLaMA
from lit_llama.tokenizer import Tokenizer
from lit_llama.loraMultiForward import lora
from lit_llama.utils import lazy_load, llama_model_lookup




def main(
    prompt: str = "In five years, Grant will be 2/3 the age of the hospital that he is hired into. If Grant is currently 25 years old, how old is the hospital now? \
          [NewPrompt] If Dan is learning to screen-print t-shirts and in the first hour he makes one t-shirt every 12 minutes, and in the second hour, he makes one every 6 minutes, how many t-shirts does he make in total over the course of those two hours? \
          [NewPrompt] When four positive integers are divided by $11$, the remainders are $2,$ $4,$ $6,$ and $8,$ respectively.\n\nWhen the sum of the four integers is divided by X$, what is the remainder?\nIf we know the answer to the above question is 9, what is the value of unknown variable X? \
          [NewPrompt] If there are 250 days per year on planet Orbius-5, and each year is divided into 5 seasons, and an astronaut from Earth stays on Orbius-5 for 3 seasons before returning to Earth, what is the total number of days the astronaut will spend on Orbius-5? \
          ",
    input: str = "",
    lora_path: Path = Path("out/lora/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0001_bs64_epoch10/epoch-7.5-valloss0.4835.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 200,
    top_k: int = 200,
    instruct_style: str = "metaMath", # or "alpaca"
    temperature: float = 0.4,
    lora_r = 128,
    lora_alpha = 16,
    lora_dropout = 0.05,
    num_of_forwards = 1,
    save_file = 'single',
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
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
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    with open("data/metaMath/test.json", 'r') as f:
        ori_prompts = json.load(f)

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)
            model.config.num_of_forwards = num_of_forwards

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)
    if ori_prompts is None:
        prompts = prompt.split("[NewPrompt]")
    else:
        prompts = [x['query'] for x in ori_prompts]
        answers = [x['response'] for x in ori_prompts]
    inputs = input.split("[NewPrompt]")
    if len(prompts) == len(inputs):
        samples = [{"instruction": p.strip(), "input": i.strip()} for p, i in zip(prompts, inputs)]
    else:
        samples = [{"instruction": p.strip(), "input": ""} for p in prompts]

    cnt = 0
    save_dict = []
    for sample in tqdm(samples[:5]):
        prompt = generate_prompt(sample, instruct_style)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)

        save_dict.append({'question': sample, 'answer': answers[cnt], 'completion': ' '.join(output.split('[|AI Assistant|]')[1:])})
        print('----------question-------')
        print(sample)
        print('----------true-----------')
        print(save_dict[-1]['answer'])
        print('----------predicted-----------')
        print(save_dict[-1]['completion'])
        cnt += 1
        # tokens_generated = y.size(0) - prompt_length
        # print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        # if fabric.device.type == "cuda":
        #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    
    with open(f'{save_file}_temp.p', 'wb') as f:
        pickle.dump(save_dict, f)

def process_results(doc, completion, answer):
    answer = answer.split('The answer is:')[1].strip()
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[1]
        extract_ans_temp = ans.split('\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        # print(extract_ans, answer)
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        # invalid_outputs.append(temp)
        return False

def eval(file_name='single_temp.p'):
    with open(file_name, 'rb') as f:
        results = pickle.load(f)

    total = []
    for result in results:
        res = process_results(result['question'], result['completion'], result['answer'])
        total.append(res)
    
    print('final score:', sum(total)/len(total))

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
    elif instruct_style == "orca" or instruct_style == "metaMath":
        return "[|User|] "+ example['instruction']+" [|AI Assistant|] "
    elif instruct_style == "beaver":
        return "[User] "+ example['instruction']+" [Assistant] "
    elif instruct_style == 'hh':
        return "Human: " + example['instruction'] + " Assistant: "
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
    # CLI(eval)
