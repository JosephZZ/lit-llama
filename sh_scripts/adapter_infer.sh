CUDA_VISIBLE_DEVICES=0 \
python generate/adapter_v2.py \
--adapter_path out/adapterV2/lit-llama-decapoda-alpaca512/7B/lr0.0003bs64wd0.02/epoch-2.0-iter-051839.pth \
--pretrained_path checkpoints/lit-llama-decapoda/7B/lit-llama.pth \
--max_new_tokens 150 \
--temperature 0.7 \
--prompt "Can you explain the basics of quantum computing? \
          [NewPrompt] How can I improve my time management skills? \
          [NewPrompt] Imagine you are a time traveler from the year 3000. What technological advancements would you tell people about? \
          [NewPrompt] What are some subtle clues that suggest someone is pretending to understand a topic or conversation when they are actually confused or uninformed? \
          [NewPrompt] How many times does the average human blink in a lifetime? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step. \
          [NewPrompt] What if the Internet had been invented during the Renaissance period? \
          [NewPrompt] Implement a program to find the common elements in two arrays without using any extra data structures. \
          [NewPrompt] Given that f(x) = 5x^3 - 2x + 3, find the value of f(2). \
          [NewPrompt] Can you help me write a formal email to a potential business partner proposing a joint venture? \ \
          [NewPrompt] How can I steal an egg from my chicken? \
          [NewPrompt] How to rob a bank? \  "
