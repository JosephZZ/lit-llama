import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main(
    prompt: str = " The following are multiple choice questions (with answers) about clinical knowledge. Pick the right choice. \n The energy for all forms of muscle contraction is provided by:",
    input: str = '''The choices are: ["ATP.","ADP.","phosphocreatine.","oxidative phosphorylation."]''',
    adapter_path: Path = Path("out/aligner/lit-llama-2-metaMath/7B/1000vector-start_layer2-lr0.009bs64weightDecay0.02wu1/epoch-3.0-valloss0.7162"),
    mode = "lora",
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    model_size = "7B",
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.1,
    instruct_style="alpaca"
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
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
    
    if mode != "raw_base":
        assert adapter_path.is_file()
        model_eval_result_path = str(adapter_path).replace(".pth", "_MMLU_eval_temp.json")
    else:
        model_eval_result_path = str(pretrained_path).replace(".pth", "_MMLU_eval_temp.json")

    data_path = 'data/MMLU/test.json'
    with open(data_path,"r+", encoding="utf8") as file:
        data = file.readlines()
        data = [json.loads(line) for line in data]

    ## load model
    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    if mode == "aligner":
        from lit_llama.aligner import LLaMA, LLaMAConfig
        config = LLaMAConfig.from_name(model_size)
        config.adapter_prompt_length = 1000
        config.adapter_start_layer = 2

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA(config)

    elif mode == "adapter":
        from lit_llama.adapter import LLaMA, LLaMAConfig
        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(model_size)

    elif mode == "lora":
        from lit_llama.model import LLaMA, LLaMAConfig
        from lit_llama.lora import  lora
        config = LLaMAConfig.from_name(model_size)
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA(config)

    elif mode == "raw_base":
        from lit_llama.model import LLaMA, LLaMAConfig
        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(model_size)

    with lazy_load(pretrained_path) as pretrained_checkpoint:
        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        if mode != "raw_base":
            with lazy_load(adapter_path) as adapter_checkpoint:
                # 2. Load the fine-tuned adapter weights
                model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)


    ## start evaluating
    tokenizer = Tokenizer(tokenizer_path)
 
    eval_results = []
    total_correct_count = 0
    for i, sample in enumerate(data):
        prompt = generate_prompt(sample, mode)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        # output = output.split("### Response:")[1].strip()
        print(output)

        model_answer = output.split("### Response: \nThe correct answer is:")[1].split("###")[0].strip().strip(".").strip("'").strip('.').strip("'")
        sample['model_answer'] = model_answer
        correct_answer = sample["choices"][sample["answer"]].strip("[").strip("]").strip().strip(".").strip("'").strip('.').strip("'")
        if model_answer == correct_answer:
            sample['is_model_correct'] = True
            total_correct_count += 1
        else:
            sample['is_model_correct'] = False

        eval_results.append(sample)
        print(f"eval results so far: {total_correct_count}/{i}")

        with open(model_eval_result_path,"w+", encoding="utf8") as file:
            for line in eval_results:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
        # tokens_generated = y.size(0) - prompt_length
        # print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        # if fabric.device.type == "cuda":
        #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    total_data = len(data)
    print(f"Total correct count: {total_correct_count}/{total_data}")
    model_eval_result_path = adapter_path.replace(".pth", f"_MMLU_eval_{total_correct_count}/{total_data}.json")
    with open(model_eval_result_path,"w+", encoding="utf8") as file:
        for line in eval_results:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

def generate_prompt(example, mode):
    if mode != "raw_base":
        return(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction: The following are multiple choice questions (with answers) about {example['subject']}.\n{example['question']}\n "
                f"Pick the correct answer from the choice list. \n### Input: The choices are: \n{example['choices']}\n\n### Response: \nThe correct answer is:"
        )
    else:        
        return(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction: The following are multiple choice questions (with answers) about {example['subject']}.\n{example['question']}\n "
                f"Pick ONE correct answer from the choice list, and answer with ONLY the choice you pick and NOTHING else. \n### Input: The choices are: \n{example['choices']}\n\n### Response: \nThe single correct answer is:"
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
