"""
Instruction-tuning with LLaMA-Adapter on the Alpaca dataset following the paper

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

This script runs on a single GPU by default. You can adjust the `micro_batch_size` to fit your GPU memory.
You can finetune within 1 hour as done in the original paper using DeepSpeed Zero-2 on 8 A100 GPUs by setting the
devices variable to `devices = 8` and `micro_batch_size = 8` (or higher).

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import sys
import time
from pathlib import Path
import shutil
from datetime import datetime
import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.rotation_outside_attention_aligner import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy

#4096960 - 1000tokens

# devices = 1

# # Hyperparameters
# learning_rate = 9e-3
# batch_size = 32 // devices
# micro_batch_size = 4
# gradient_accumulation_iters = batch_size // micro_batch_size
# assert gradient_accumulation_iters > 0
# epoch_size = 100  # train dataset size ; alpaca is 50000, isotonic is 280000, baize is 200000
# num_epochs = 200
# max_iters = int(num_epochs * (epoch_size // micro_batch_size) // devices)
# weight_decay = 0.02
# max_seq_length = 512 #alpaca 256, dolly 1024, lima 2048, isotonic 1536 # see scripts/prepare_alpaca.py
# warmup_epoch = 15
# warmup_iters = warmup_epoch * (epoch_size // micro_batch_size) // devices  # 2 alpaca epochs
# start_iter = int(0)

# instruction_tuning = True
# eval_interval = 1 * epoch_size // batch_size
# save_interval = 1 * epoch_size // batch_size
# eval_iters = 30
# log_interval = 10

# ds_config = {
#     "train_micro_batch_size_per_gpu": micro_batch_size,
#     "gradient_accumulation_steps": gradient_accumulation_iters,
#     "zero_optimization": {"stage": 2},
# }

# safer_or_better = 'safer'
# ##****###
# model_size = '7B'
# aligner_length = 1
# data_dir = Path("data/rolebench/eng_instruction_generalization_sheldon_100")
# ##****###
# aligner_start_layer = 2
# model_base = "lit-llama-2"
# model_version = model_size
# pretrained_path = Path(f"checkpoints/{model_base}/{model_version}/lit-llama.pth")
# previous_aligner_path = ""
# previous_optimizer_path = ""
# out_dir = Path(f"out/aligner/{model_base}-{data_dir.name}/{model_version}/{aligner_length}vector-start_layer{aligner_start_layer}-lr{learning_rate}bs{int(batch_size)}weightDecay{weight_decay}wu{warmup_epoch}/")
# save_model_name = f"{model_base}-{model_version}-{aligner_length}vector-start_layer{aligner_start_layer}-lr{learning_rate}bs{int(batch_size)}wu{warmup_epoch}.pth"

# print(f"Training LLaMA-Aligner with base model {model_base} using {model_version} parameters on the {data_dir.name} dataset, saving to {out_dir}")

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# put the above hyperparameters inside the main function inputs
def main(
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        micro_batch_size: int = 4,
        epoch_size: int = 100,
        num_epochs: int = 500,
        weight_decay: float = 0.02,
        max_seq_length: int = 512,
        warmup_epoch: int = 30,
        start_iter: int = 0,
        instruction_tuning: bool = True,
        eval_interval: int = 30,
        save_interval: int = 30,
        eval_iters: int = 30,
        log_interval: int = 10,
        safer_or_better: str = 'safer',
        model_size: str = '7B',
        aligner_length: int = 1,
        data_dir: Path = Path("data/rolebench/eng_instruction_generalization_sheldon_100"),
        aligner_start_layer: int = 2,
        model_base: str = "lit-llama-2",
        model_version: str = "7B",
        devices: int = 1,
        previous_aligner_path: str = "", #"out/rotation_before_scale_aligner_globalRotation-False/lit-llama-2-eng_instruction_generalization_sheldon_100/7B/1vector-start_layer2-lr0.005bs32weightDecay0.02wu30/final.pth",
        previous_optimizer_path: str = "", #"out/rotation_before_scale_aligner_globalRotation-False/lit-llama-2-eng_instruction_generalization_sheldon_100/7B/1vector-start_layer2-lr0.005bs32weightDecay0.02wu30/optimizer-final-epoch-600.0--valloss5.7226.pth",
        using_global_rotation: bool = False,
        rotation_place = "normedX_to_mlp",
):
    
    pretrained_path: Path = Path(f"checkpoints/{model_base}/{model_version}/lit-llama.pth")
    out_dir: Path = Path(f"out/rotation_outside_attention_aligner/rotate_{rotation_place}_globalRotation-{using_global_rotation}/{model_base}-{data_dir.name}/{model_version}/{aligner_length}vector-start_layer{aligner_start_layer}-lr{learning_rate}bs{int(batch_size)}weightDecay{weight_decay}wu{warmup_epoch}/")
    max_iters: int = int(num_epochs * (epoch_size // micro_batch_size) // devices)
    save_model_name = f"{model_base}-{model_version}-{aligner_length}vector-start_layer{aligner_start_layer}-lr{learning_rate}bs{int(batch_size)}wu{warmup_epoch}.pth"
    warmup_iters = warmup_epoch * (epoch_size // micro_batch_size) // devices  # 2 alpaca epochs

    print(f"Training LLaMA-Aligner with base model {model_base} using {model_version} parameters on the {data_dir.name} dataset, saving to {out_dir}")

    gradient_accumulation_iters = batch_size // micro_batch_size
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_iters,
        "zero_optimization": {"stage": 2},
    }
    
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy=(DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto"), 
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)


    def train(
        fabric: L.Fabric,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: np.ndarray,
        val_data: np.ndarray,
        out_dir: str,
       
    ) -> None:
        """The training loop.

        Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
        """
        step_count = 0

        for iter_num in range(start_iter,max_iters):
            epoch = iter_num * micro_batch_size * devices / epoch_size
            if step_count <= warmup_iters:
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
                    val_loss = validate(fabric, model, val_data)
                    fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                    fabric.barrier()
                    with open(os.path.join(out_dir, "val log.txt"), "a") as file:
                        file.write(f"iter {iter_num}: val loss {val_loss:.6f}\n")

                if step_count % save_interval == 0:
                    print(f"Saving adapter weights to {out_dir}")
                    # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                    aligner_path = os.path.join(out_dir, f"epoch-{epoch:.1f}-valloss{val_loss:.4f}")
                    optimizer_path = os.path.join(out_dir, f"optimizer-epoch-{epoch:.1f}-valloss{val_loss:.4f}.pth")
                    save_model_checkpoint(fabric, model, optimizer, aligner_path, optimizer_path)

            dt = time.time() - t0
            if iter_num % log_interval == 0:
                fabric.print(f"epoch-{epoch:.1f} iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, model_name:{out_dir}")
                with open(os.path.join(out_dir, "train_log.txt"), "a") as file:
                    file.write(f"epoch-{epoch:.1f} iter {iter_num}: train loss {loss.item():.6f} {datetime.now()}\n")
            # Save the final checkpoint at the end of training
        aligner_path = os.path.join(out_dir, f"final.pth")
        optimizer_path = os.path.join(out_dir, f"optimizer-final-epoch-{epoch:.1f}--valloss{val_loss:.4f}.pth")
        save_model_checkpoint(fabric, model,  optimizer, aligner_path, optimizer_path)


    def generate_response(model, instruction, input=""):
        tokenizer = Tokenizer("checkpoints/lit-llama-2/tokenizer.model")
        sample = {"instruction": instruction, "input": input}
        prompt = instruction
        if instruction_tuning:
            prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

        output = generate(
            model,
            idx=encoded,
            max_seq_length=max_seq_length,
            max_new_tokens=100,
            temperature=0.8,
        )
        output = tokenizer.decode(output)
        return output # output.split("### Response:")[1].strip()


    @torch.no_grad()
    def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
        fabric.print("Validating ...")
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, targets = get_batch(fabric, val_data)
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            losses[k] = loss.item()
        val_loss = losses.mean()

        # produce an example:
        instruction = "The goal of this task is to determine whether a given expression is valid or not according to the rules of arithmetic. An expression is valid if it follows all the rules of arithmetic (i.e., it does not contain any division by zero, and all exponents are non-negative integers).\nExpression: 8 / 0."
        output = generate_response(model, instruction)
        fabric.print(instruction)
        fabric.print(output)

        with open(os.path.join(out_dir, "log.txt"), "a") as file:
            file.write(f"\n###Instruction\n{instruction}\n###Response\n{output}\n\n")

        model.train()
        return val_loss.item()

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


        max_len = max(len(s) for s in input_ids)

        def pad_right(x, pad_id):
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


    def save_model_checkpoint(fabric, model, optimizer, aligner_path, optimizer_path):
        aligner_path = Path(aligner_path)

        if isinstance(fabric.strategy, DeepSpeedStrategy):
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

            tmp_path = aligner_path.with_suffix(".tmp")
            fabric.save(tmp_path, {"model": model})
            fabric.barrier()
            if fabric.global_rank == 0:
                # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
                # and only keep the adapter weights
                state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
                state_dict = adapter_state_from_state_dict(state_dict)
                torch.save(state_dict, aligner_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                shutil.rmtree(tmp_path)
        else:
            state_dict = adapter_state_from_state_dict(model.state_dict())
            if fabric.global_rank == 0:
                torch.save(state_dict, aligner_path)
                torch.save(optimizer.state_dict(), optimizer_path)
            fabric.barrier()



    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name(model_size)
    config.block_size = max_seq_length
    config.using_global_rotation = using_global_rotation
    config.adapter_start_layer = aligner_start_layer
    config.rotation_place = rotation_place

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path)

    with fabric.init_module():
        model = LLaMA(config)
        # strict=False because missing keys due to adapter weights not containted in state dict
        model.load_state_dict(checkpoint, strict=False)
        if previous_aligner_path:
            model.load_state_dict(torch.load(previous_aligner_path), strict=False)

    # print("aligner length: ",model.config.adapter_prompt_length)

    mark_only_adapter_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if previous_optimizer_path:
        optimizer.load_state_dict(torch.load(previous_optimizer_path))

    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, out_dir)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)