import matplotlib.pyplot as plt

with open('/home/shuwen/ziheng/llm/lit-llama/out/loraAllLayerMultiForward_1/lit-llama-2-metaMath/7B/lora_r128_alpha16_dropout0.05_lr0.0003_bs64_epoch10warmup4937/val log.txt', 'r') as f:
    lines = f.readlines()

x = []
y = []

for line in lines:
    words = line.split()
    for i in range(len(words)):
        if words[i] == 'epoch':
            x.append(float(words[i+1]))
        if words[i] == 'loss':
            y.append(float(words[i+1]))
            break

assert len(x) == len(y)
plt.plot(x, y)
plt.savefig('temp.png')