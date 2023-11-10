"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
from datetime import datetime

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.enableLoraAllLayer import enable_lora, mark_only_lora_as_trainable, lora_state_dict
from lit_llama.modelMultiForward import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#293665280

instruction_tuning = True
devices = 1

# Hyperparameters
epoch_size = 395000
learning_rate = 3e-4
batch_size = 64
micro_batch_size = 2
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_num = 10
max_iters = epoch_size * epoch_num // micro_batch_size
weight_decay = 0.02
max_seq_length = 1536  # see scripts/prepare_alpaca.py
lora_r = 128
lora_alpha = 16
lora_dropout = 0.05
warmup_epoch = 0.05
warmup_iters = int (warmup_epoch * (epoch_size // micro_batch_size) // devices)  # 2 alpaca epochs

#***************#
num_of_forwards = 2
#***************#

data_name = "metaMath"
data_dir: str = Path(f"data/{data_name}")
model_base = "lit-llama-2"
model_version = "7B"
pretrained_path: str = f"checkpoints/{model_base}/{model_version}/lit-llama.pth"
tokenizer_path: str = f"checkpoints/{model_base}/tokenizer.model"
out_dir: str = f"out/loraAllLayerMultiForward_{num_of_forwards}/{model_base}-{data_name}/{model_version}/lora_r{lora_r}_alpha{lora_alpha}_dropout{lora_dropout}_lr{learning_rate}_bs{batch_size}_epoch{epoch_num}warmup{warmup_iters}"

previous_checkpoing = ""
# "out/lora/lit-llama-decapoda-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs64_epoch2/iter-051967-ckpt.pth"
# "out/lora/lit-llama-decapoda-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs32_epoch2/iter-103999-ckpt.pth"
previous_optimizer = ""
# "out/lora/lit-llama-decapoda-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs64_epoch2/optimizer-iter-051967-ckpt.pth"
#"out/lora/lit-llama-decapoda-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs32_epoch2/optimizer-iter-103999-ckpt.pth"
start_iter = 0 * epoch_size // micro_batch_size

safer_or_better = 'safer'

eval_interval =  0.1*epoch_size // batch_size
save_interval = 0.2*epoch_size // batch_size
eval_iters =  30 if "lima" in data_name else 100
log_interval = 10


print(f"Training LoRA with base model {pretrained_path} on the {data_dir.name} dataset, saving to {out_dir}")

def main(

):

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name(model_version)
    config.block_size = max_seq_length
    config.num_of_forwards = num_of_forwards
    config.lora_r = lora_r
    config.lora_alpha = lora_alpha
    config.lora_dropout = lora_dropout

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module():
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
        if previous_checkpoing:
            model.load_state_dict(torch.load(previous_checkpoing), strict=False)

        enable_lora(model, config)
        mark_only_lora_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if previous_optimizer:
        optimizer.load_state_dict(torch.load(previous_optimizer))

    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(start_iter, max_iters):
        epoch = iter_num * micro_batch_size / epoch_size

        if warmup_iters > 0 and step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)
                fabric.print(f"epoch {epoch:.1f} step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                with open(os.path.join(out_dir, "val log.txt"), "a") as file:
                    file.write(f"epoch {epoch:.3f} iter {iter_num}: val loss {val_loss:.6f}\n")

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model.state_dict())
                fabric.save(os.path.join(out_dir, f"epoch{epoch:.3f}-valloss{val_loss:.4f}.pth"), checkpoint)
                fabric.save(os.path.join(out_dir, f"epoch{epoch:.3f}-valloss{val_loss:.4f}-ckpt.pth"), optimizer.state_dict())
        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"epoch {epoch:.1f} iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, experiment: {out_dir}")
            with open(os.path.join(out_dir, "train_log.txt"), "a") as file:
                file.write(f"epoch {epoch:.1f} iter {iter_num}: train loss {loss.item():.6f}  {datetime.now()}\n")

def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, tokenizer_path)
    fabric.print(instruction)
    fabric.print(output)

    with open(os.path.join(out_dir, "log.txt"), "a") as file:
        file.write(f"Validation loss: {out:.6f}\n")
        file.write(f"\n###Instruction\n{instruction}\n###Response\n{output}\n\n")

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    if "philo" in data_dir.name or "orca" in data_dir.name or "metaMath" in data_dir.name:
        input_ids = [data[i]["dialog_ids"].type(torch.int64) for i in ix]
        labels = [data[i]["labels"].type(torch.int64) for i in ix]  
    elif "hh" in data_dir.name:
        input_ids_chosen = [data[i]["chosen"].type(torch.int64) for i in ix]
        input_ids = input_ids_chosen 
        labels = [i.clone() for i in input_ids]
    elif "beaver" in data_dir.name:
        if safer_or_better == 'safer':
            using_index = "safer_response_id"
        elif safer_or_better == "better":
            using_index = "better_response_id"
        input_ids = [ data[i][f"dialog_{data[i][using_index]}"].type(torch.int64) for i in ix]
        labels = [i.clone() for i in input_ids]
    else:
        input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
        labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = min(max(len(s) for s in input_ids), max_seq_length)
    # print ("max seq len: ", max_len)
    def pad_right(x, pad_id):
        if len(x) > max_len:
            #truncate based on max length
            return torch.tensor(x[:max_len])
        else:
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
