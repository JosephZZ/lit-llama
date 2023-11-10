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
from lit_llama.alignerCombine import LLaMA
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What is conciousness?",
    input: str = "",
    aligner_path_list = [Path("out/aligner/lit-llama-2-lima/7B/1vector-start_layer2-lr0.0001bs16/epoch-14.3-iter-007167.pth"), \
                         Path("out/aligner/lit-llama-2-orca/7B/1vector-start_layer2-lr9e-05bs32/epoch-1.3.pth"), \
                         Path("out/aligner/lit-llama-2-baize/7B/10vector-start_layer2-lr9e-05bs32/epoch-5.0-iter-049999.pth")],
    pretrained_path: Path = Path("checkpoints/lit-llama-2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama-2/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.7,
    aligner_length: int = 2,
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
    for ap in aligner_path_list:
        assert ap.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    aligner_list = []
    aligner_embd_list = []
    aligner_length = 0
    for ap in aligner_path_list:
        al = torch.load(ap)
        ae = al["global_value_embedding.weight"]
        aligner_list.append(al)
        aligner_embd_list.append(ae)
        aligner_length += ae.shape[0]

    with lazy_load(pretrained_path) as pretrained_checkpoint :
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name, aligner_length)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights

    cur_pos = 0
    last_pos = 0
    for ae in aligner_embd_list:
        cur_pos = last_pos + ae.shape[0]
        model.global_value_embedding.weight.data[last_pos:cur_pos] = ae
        last_pos = cur_pos

    for name_model, param_model in model.named_parameters():
        if "attn.gating_factor" in name_model:
            cur_pos = 0
            last_pos = 0
            for i in range(len(aligner_list)):
                for name_aligner, param_aligner in aligner_list[i].items():
                    if name_model == name_aligner:
                        if i == len(aligner_list) - 1:
                            cur_pos = None
                        else:
                            cur_pos = last_pos + param_aligner.shape[-1]
                        param_model.data[:,:,:,last_pos:cur_pos] = param_aligner
                        last_pos = cur_pos
                        break


    # adapter_matched = {name: param for name, param in adapter_checkpoint.items() if name in model_state_dict and model_state_dict[name].size() == param.size()}

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    print ("aligner_length:", model.config.adapter_prompt_length)

    tokenizer = Tokenizer(tokenizer_path)
    prompts = prompt.split("[NewPrompt]")
    inputs = input.split("[NewPrompt]")
    if len(prompts) == len(inputs):
        samples = [{"instruction": p.strip(), "input": i.strip()} for p, i in zip(prompts, inputs)]
    else:
        samples = [{"instruction": p.strip(), "input": ""} for p in prompts]

    for sample in samples:
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        # output = output.split("### Response:")[1].strip()
        print(output)

        tokens_generated = y.size(0) - prompt_length
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
