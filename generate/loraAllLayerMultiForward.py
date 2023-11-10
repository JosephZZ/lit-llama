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
from lit_llama.modelMultiForward import LLaMA
from lit_llama.tokenizer import Tokenizer
from lit_llama.enableLoraAllLayer import enable_lora, mark_only_lora_as_trainable, lora_state_dict
from lit_llama.utils import lazy_load, llama_model_lookup




def main(
    prompt: str = "In five years, Grant will be 2/3 the age of the hospital that he is hired into. If Grant is currently 25 years old, how old is the hospital now? \
          [NewPrompt] If Dan is learning to screen-print t-shirts and in the first hour he makes one t-shirt every 12 minutes, and in the second hour, he makes one every 6 minutes, how many t-shirts does he make in total over the course of those two hours? \
          [NewPrompt] When four positive integers are divided by $11$, the remainders are $2,$ $4,$ $6,$ and $8,$ respectively.\n\nWhen the sum of the four integers is divided by X$, what is the remainder?\nIf we know the answer to the above question is 9, what is the value of unknown variable X? \
          [NewPrompt] If there are 250 days per year on planet Orbius-5, and each year is divided into 5 seasons, and an astronaut from Earth stays on Orbius-5 for 3 seasons before returning to Earth, what is the total number of days the astronaut will spend on Orbius-5? \
          ",
    input: str = "",
    lora_path: Path = Path("out/loraAllLayerMultiForward_2/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup9875/epoch6.198-valloss0.1930.pth"),
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
    num_of_forwards = 2,
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

    ori_prompts = [
        {"response":"The graph of $y = |f(x)|$ is the graph of $y = f(x)$ reflected across the x-axis for all values of $x$ where $f(x) < 0$.\nTherefore, the correct letter that corresponds to the graph of $y = |f(x)|$ is the letter $\\boxed{\\text{D}}$.The answer is: \\text{D}","query":"Which letter corresponds to the graph of $y = |f(x)|$?","type":"MATH_Rephrased"},
        {"response":"If the coffee shop sold 6 teas, then the number of lattes sold is 8 more than four times the number of teas.\nFour times the number of teas is 4 * 6 = 24.\nSo, the number of lattes sold is 24 + 8 = 32.\n#### 32\nThe answer is: 32","query":"The coffee shop sold 8 more than four times the number of lattes as it did teas. If 6 teas were sold how many lattes did the coffee shop sell?","type":"GSM_AnsAug"},
        {"response":"The annual profit is $8,000.\nThe sum of the profits in the first, third, and fourth quarters is $1,500 + $3,000 + $2,000 = $6,500.\nTherefore, the profit in the second quarter is $8,000 - $6,500 = $1,500.\n#### 1500\nThe answer is: 1500","query":"A store is comparing their profits throughout the year. They had profits of $1,500 in the first quarter of the year, $3,000 in the third quarter, and $2,000 in the fourth quarter. If their annual profits are $8,000, how much profit, in dollars, did they make in the second quarter?","type":"GSM_AnsAug"},
        {"response":"To solve this problem, we need to determine the value of x, which represents the number of students who said they liked chocolate cake.\nWe know that 18 students liked apple pie, 12 students did not like either, and 5 students liked both.\nLet's set up the equation using the principle of inclusion-exclusion:\nNumber of students who liked apple pie + Number of students who liked chocolate cake - Number of students who liked both + Number of students who liked neither = Total number of students\n18 + x - 5 + 12 = 40\nLet's simplify and solve for x:\n18 + x - 5 + 12 = 40\n25 + x = 40\nTo isolate x, we subtract 25 from both sides of the equation:\n25 + x - 25 = 40 - 25\nx = 15\nThe value of x is 15.\nThe answer is: 15","query":"In a class of $40$ students, $18$ said they liked apple pie, X said they liked chocolate cake, and $12$ said they did not like either. 5 students in the class liked both. What is the value of unknown variable X?","type":"MATH_SV"},
        {"response":"Finishing her homework takes 30 minutes.\nCleaning her room takes half as long as her homework, so it takes 30\/2 = 15 minutes.\nWalking the dog takes 5 minutes more than doing homework, so it takes 30 + 5 = 35 minutes.\nTaking out the trash takes 1\/6 of the time it takes her to do homework, so it takes 30\/6 = 5 minutes.\nThe total time for all the tasks is 30 + 15 + 35 + 5 = 85 minutes.\nJanice has 2 hours, which is 2 * 60 = 120 minutes.\nSo the remaining time Janice has before the movie starts is 120 - 85 = 35 minutes.\n#### 35\nThe answer is: 35","query":"With Janice needing to watch a movie in 2 hours, she has several tasks to complete before she can leave. These tasks include finishing her homework (which takes 30 minutes), cleaning her room (which takes half as long as her homework), walking the dog (which takes 5 minutes more than doing homework), and taking out the trash (which takes 1\/6 of the time it takes her to do homework). What is the remaining time in minutes that Janice has before the movie starts?","type":"GSM_Rephrased"},
        {"response":"The area of one side of the roof is 20 feet x 40 feet = 800 square feet\nSince there are two sides to each roof, the total area of one roof is 800 square feet x 2 = 1600 square feet\nJerry needs 8 shingles to cover 1 square foot, so to cover 1600 square feet he will need 1600 square feet x 8 shingles\/square foot = 12800 shingles\nSince he has to shingle 3 roofs, the total number of shingles he needs is 12800 shingles x 3 roofs = 38400 shingles\n#### 38400\nThe answer is: 38400","query":"Jerry has to shingle 3 roofs. Each roof consists of two slanted rectangular sides that measure 20 feet by 40 feet. If he requires 8 shingles to cover one square foot of roof, what is the total number of shingles he needs?","type":"GSM_Rephrased"},
        {"response":"There are 60 minutes in 1 hour, so 2 hours is equal to 2 x 60 = 120 minutes.\nIf Bob can shuck 10 oysters in 5 minutes, then in 120 minutes he can shuck 120\/5 = 24 sets of 10 oysters.\nSo, Bob can shuck 24 x 10 = 240 oysters in a span of 2 hours.\n#### 240\nThe answer is: 240","query":"If Bob can shuck 10 oysters in 5 minutes, how many oysters can he shuck in a span of 2 hours?","type":"GSM_Rephrased"},
        {"response":"The area of one 20 x 15 foot lawn is 20 x 15 = 300 square feet.\nLaKeisha has already mowed three lawns, so she has mowed 3 x 300 = 900 square feet.\nThe cost of the book set is $150, and she charges $.10 per square foot, so she needs to mow 150 \/ 0.10 = 1500 square feet to earn enough for the book set.\nShe has already mowed 900 square feet, so she needs to mow an additional 1500 - 900 = 600 square feet.\n#### 600\nThe answer is: 600","query":"LaKeisha is mowing lawns to raise money for a collector set of books. She charges $.10 for every square foot of lawn. The book set costs $150. If she has already mowed three 20 x 15 foot lawns, how many more square feet does she have to mow to earn enough for the book set?","type":"GSM_AnsAug"},
        {"response":"To solve this problem, we need to determine the value of x, which represents the letter that has already been typed.\nWe are given that letter 8 has already been typed. Therefore, x is equal to 8.\nThe value of x is 8.\nThe answer is: 8","query":"In an office at various times during the day, the boss gives the secretary a letter to type, each time putting the letter on top of the pile in the secretary's in-box. When there is time, the secretary takes the top letter off the pile and types it. There are nine letters to be typed during the day, and the boss delivers them in the order $1, 2, 3, 4, 5, 6, 7, 8, 9$.\nWhile leaving for lunch, the secretary tells a colleague that letter $8$ has already been typed, but says nothing else about the morning's typing. The colleague wonders which of the nine letters remain to be typed after lunch and in what order they will be typed. Based upon the above information, how many such after-lunch typing orders are possible? (That there are no letters left to be typed is one of the possibilities.)\nRe-stating the problem for clarity, let $S$ be a set arranged in increasing order. At any time an element can be appended to the end of $S$, or the last element of $S$ can be removed. The question asks for the number of different orders in which the all of the remaining elements of $S$ can be removed, given that X had been removed already. The answer is 704. What is the value of unknown variable X?","type":"MATH_SV"},
        {"response":"In the second month, Grant made 350*2+50 = $750\nThe sum of the first two months is 350+750 = $1100\nIn the third month, Grant made 1100*4 = $4400\nTherefore, Grant made a total of 350+750+4400 = $5500 in his first three months.\n#### 5500\nThe answer is: 5500","query":"Grant has worked as a freelance math worker for the last three months.  The first month he made 350$.  The second month he made 50$ more than double he made the first month.  The third month he quadrupled the sum of the first two months.  How much did Grant make in his first three months total?","type":"GSM_AnsAug"},
    ]

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True):
            model = LLaMA.from_name(name)
            model.config.lora_r = lora_r
            model.config.lora_alpha = lora_alpha
            model.config.lora_dropout = lora_dropout
            model.config.num_of_forwards = num_of_forwards

            enable_lora(model, model.config)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    ori_prompts=None
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
    for sample in samples:
        prompt = generate_prompt(sample, instruct_style)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        print(output)

        # output = output.split("### Response:")[1].strip()
        if ori_prompts is not None:
            print('--------true answer------')
            print(answers[cnt].lower().split('the answer is')[1])
            print('--------------------------')
            print('--------------------------')
            cnt += 1

        tokens_generated = y.size(0) - prompt_length
        print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

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
