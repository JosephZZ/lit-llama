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

promptAligner_version = "V8NoBase"
from lit_llama.promptAlignerV8NoBase import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_home_dir = Path("/home/shuwen/ziheng/llm/lit-llama")

def main(
    prompt: str = "What is the relationship between morality and law?",
    input: str = "",
    system_align_prompt: str = "Michel Foucault (1926-1984 CE): A French philosopher, known for his critical studies of social institutions, including mental institutions, prisons, and the sciences.\nYou are an assistant taking on the perspective and value of this person.",
    aligner_path: Path = project_home_dir / "out/promptAlignerV4cat/lit-llama-2-orca/7B/2vector-start_layer2-lr1e-05bs16/epoch-1.0.pth",
    pretrained_path: Path = project_home_dir / "checkpoints/lit-llama-2/7B/lit-llama.pth",
    tokenizer_path: Path = project_home_dir / "checkpoints/lit-llama-2/tokenizer.model",
    initial_embedding_base_path: Path = project_home_dir / "out/aligner/lit-llama-2-baize/7B/1vector-start_layer2-lr9e-05bs64/epoch-2.55967-iter-255967.pth",
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.7,
    aligner_length: int = 10,
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
    assert aligner_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)


    tokenizer = Tokenizer(tokenizer_path)
    prompts = prompt.split("[NewPrompt]")
    system_align_prompts = system_align_prompt.split("[NewPrompt]")
    print(prompts)
    print(system_align_prompts)
    if len(prompts) != len(system_align_prompts):
        raise ValueError(f"The number of prompts ({len(prompts)}) and \
                         the number of system_align_prompts ({len(system_align_prompts)}) are not same.")


    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(aligner_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)

        model_state_dict = model.state_dict()

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        # Create a new state dictionary with matching parameters
        # adapter_matched = {name: param 
        #                    if (name in model_state_dict and model_state_dict[name].size() == param.size()) 
        #                    else param.reshape(model_state_dict[name].size()) for name, param in adapter_checkpoint.items() if name in model_state_dict}
        adapter_matched = {name: param for name, param in adapter_checkpoint.items() if name in model_state_dict and model_state_dict[name].size() == param.size()}
        model.load_state_dict(adapter_matched, strict=False)

        # if initial_embedding_base_path:
        #     embedding_base = torch.load(initial_embedding_base_path)["global_value_embedding.weight"] 
        #     model.pretrained_aligner_embedding_base.data = embedding_base.unsqueeze(0) 
        #     model.pretrained_aligner_embedding_base.requires_grad = False

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    print ("config:", model.config)

    for prompt, system_align_prompt in zip(prompts, system_align_prompts):
        prompt = "[|User|]" + prompt.strip() + "[|AI Assistant|]" 
        prompt_encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = prompt_encoded.size(0)
        system_align_prompt_encoded = tokenizer.encode(system_align_prompt.strip(), bos=True, eos=False, device=model.device)
        system_align_prompt_encoded = torch.stack([system_align_prompt_encoded])
        t0 = time.perf_counter()
        model.set_model_mode(is_aligner=False)
        aligner_embedding, _ = model(system_align_prompt_encoded) #suppose we use bf16, otherwise change it
        print("aligner_embedding:", aligner_embedding)
        t1 = time.perf_counter()
        model.set_model_mode(is_aligner=True, aligner_embedding=aligner_embedding.to(torch.bfloat16))
        y = generate(model, prompt_encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t1

        model.reset_cache()
        output = tokenizer.decode(y)
        # output = output.split("### Response:")[1].strip()
        print(output)

        tokens_generated = y.size(0) - prompt_length
        print(f"\n\nTime for generating aligner embedding: {t1 - t0:.02f} sec", file=sys.stderr)
        print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
