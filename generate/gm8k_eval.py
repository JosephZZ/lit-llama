import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import jsonlines
import lightning as L
import torch
import pickle
from tqdm import tqdm
import json
from utils import is_equiv, last_boxed_only_string

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.tokenizer import Tokenizer


from lit_llama.utils import lazy_load, llama_model_lookup, quantization
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# checkpoint example (you can choose others)
# single forward： out/loraAllLayerMultiForward_1/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup4937/epoch9.997-valloss0.1871.pth
# double： out/loraAllLayerMultiForward_2/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup9875/epoch7.198-valloss0.1894.pth

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def main(
    prompt: str = "In five years, Grant will be 2/3 the age of the hospital that he is hired into. If Grant is currently 25 years old, how old is the hospital now? \
          [NewPrompt] If Dan is learning to screen-print t-shirts and in the first hour he makes one t-shirt every 12 minutes, and in the second hour, he makes one every 6 minutes, how many t-shirts does he make in total over the course of those two hours? \
          [NewPrompt] When four positive integers are divided by $11$, the remainders are $2,$ $4,$ $6,$ and $8,$ respectively.\n\nWhen the sum of the four integers is divided by X$, what is the remainder?\nIf we know the answer to the above question is 9, what is the value of unknown variable X? \
          [NewPrompt] If there are 250 days per year on planet Orbius-5, and each year is divided into 5 seasons, and an astronaut from Earth stays on Orbius-5 for 3 seasons before returning to Earth, what is the total number of days the astronaut will spend on Orbius-5? \
          ",
    input: str = "",
    adapter_path: Path = Path("out/loraAllLayerMultiForward_2/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup9875/epoch7.198-valloss0.1894.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    mode = "loraAllLayerMultiForward",
    model_size = "7B",
    eval_file = "GSM8K", # "GSM8K" or "MATH" or customized jsonl file
    aligner_length = 1000,
    quantize: Optional[str] = None,
    range_start: int = 0,
    range_end: Optional[int] = None,
    max_new_tokens: int = 200,
    top_k: int = 200,
    num_of_forwards: int=2,
    instruct_style: str = "metaMath", # or "alpaca"
    temperature: float = 0.4,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained LoRA weights, which are the output of
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
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    prompts, answers = [], []
    if eval_file == "GSM8K":
        data_path = 'data/metaMath/GSM8K_test.jsonl'
    elif eval_file == "MATH":
        data_path = "data/metaMath/MATH_test.jsonl"
    else:
        data_path = eval_file
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if "GSM8K" in data_path:
                temp_instr = instruction=item["query"]
                temp_ans = item['response'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
                prompts.append(temp_instr)
                answers.append(temp_ans)
                eval_set = 'gm8k'
            elif "MATH" in data_path:
                temp_instr = item["instruction"]
                prompts.append(temp_instr)
                solution = item['output']
                temp_ans = remove_boxed(last_boxed_only_string(solution))
                answers.append(temp_ans)
                eval_set = 'math'

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    if mode == "aligner":
        from lit_llama.aligner import LLaMA, LLaMAConfig
        config = LLaMAConfig.from_name(model_size)
        config.adapter_prompt_length = aligner_length
        config.adapter_start_layer = 2

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA(config)

    elif mode == "adapter":
        from lit_llama.adapter import LLaMA, LLaMAConfig

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(model_size)

    elif mode == "lora":
        from lit_llama import LLaMA
        from lit_llama.lora import lora

        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(model_size)

    elif mode == "loraAllLayerMultiForward":
        from lit_llama.modelMultiForward import LLaMA
        from lit_llama.loraMultiForward import lora
        from lit_llama.enableLoraAllLayer import enable_lora

        with fabric.init_module(empty_init=True):
            model = LLaMA.from_name(model_size)
            model.config.lora_r = 128
            model.config.lora_alpha = 16
            model.config.lora_dropout = 0.05
            model.config.num_of_forwards = num_of_forwards
            enable_lora(model, model.config)

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)
    if data_path is None:
        prompts = prompt.split("[NewPrompt]")

    samples = [{"instruction": p.strip(), "input": ""} for p in prompts]

    cnt = 0
    correct_cnt = 0
    save_dict = []

    save_file = str(adapter_path) + f"_{eval_set}_eval_results.json"
    if os.path.exists(save_file):
        with open(f'{save_file}', 'r') as f:
            save_dict = json.load(f)
        range_start = len(save_dict)
        print(range_start)

    print(len(samples))
    for sample in tqdm(samples[range_start:range_end]):
        prompt = generate_prompt(sample, instruct_style)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        completion = ' '.join(output.split('[|AI Assistant|]')[1:])

        is_model_correct, extracted_ans = process_results(sample, completion, answers[cnt])
        save_dict.append({'question': sample, 'answer': answers[cnt], 'completion': completion, 'extracted_ans': extracted_ans, 'is_model_correct': is_model_correct})
        
        cnt += 1
        correct_cnt += is_model_correct

        print('----------question-------')
        print(sample)
        print('----------output-----------')
        print(completion)
        print('----------true-----------')
        print(save_dict[-1]['answer'])
        print('----------predicted-----------')
        print(extracted_ans)
        print('----------total correct so far-----------')
        print(correct_cnt, ' / ', cnt)
        # tokens_generated = y.size(0) - prompt_length
        # print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        # if fabric.device.type == "cuda":
        #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    
    save_file = str(adapter_path) + f"_{eval_set}_eval_results.json"
    with open(f'{save_file}', 'w') as f:
        json.dump(save_dict, f, indent=4)


def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    answer = str(answer)
    if len(split_ans) > 1:
        ans = split_ans[1]
        extract_ans_temp = ans.split("#")[0].split('\n')[0]
        extract_ans = extract_ans_temp.strip().strip('.').strip().strip('.').strip()
        if is_equiv(extract_ans, answer):
            return True, extract_ans
        else:
            return False, extract_ans
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        # invalid_outputs.append(temp)
        return False, None

def eval(file_path='/home/shuwen/ziheng/llm/lit-llama/out/loraAllLayerMultiForward_1/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup4937/epoch9.997-valloss0.1871.pth_test.json'):
    with open(file_path, 'r') as f:
        results = json.load(f)
    print(len(results))
    total = []
    for result in results:
        res,_ = process_results(result['question'], result['completion'], result['answer'])
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
