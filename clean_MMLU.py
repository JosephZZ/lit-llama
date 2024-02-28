import json

data_path = 'out/lora/lit-llama-2-alpaca512/7B/lora_r8_alpha16_dropout0.05_lr0.0003_bs64_epoch5/epoch5-iter-032479-ckpt_MMLU_eval_temp.json'
with open(data_path,"r+", encoding="utf8") as file:
    data = file.readlines()
    data = [json.loads(line) for line in data]

total_correct_count = 0
new_samples = []
longest_answer = 0
for i, sample in enumerate(data):
    correct_answer = sample["choices"][sample["answer"]].strip("[").strip("]").strip("'").strip('.').strip("`").strip("'").strip('.').strip("`")
    model_answer = sample["model_answer"].strip("[").strip("]").strip("'").strip('.').strip("`").strip("'").strip('.').strip("`")
    
    longest_choice = max([len(c) for c in sample["choices"]])
    if len(model_answer) > longest_choice:
        print(i,"correct: ", sample["choices"][sample["answer"]], " model: ", model_answer, )
        sample["is_model_correct"]=False
        new_samples.append(sample)

        continue
    
    sample["model_answer"] = model_answer
    if len(model_answer) > longest_answer:
        longest_answer = len(model_answer)
    if len(correct_answer) > longest_answer:
        longest_answer = len(correct_answer)
    
    shorter_of_two = min(len(correct_answer),len(model_answer))
    if shorter_of_two>0 and correct_answer[:shorter_of_two] == model_answer[:shorter_of_two]:
        total_correct_count += 1
        sample["is_model_correct"]=True
    else:
        sample["is_model_correct"]=False
    new_samples.append(sample)

print(total_correct_count,len(data))
assert(len(new_samples) == len(data))
with open(data_path,"w+", encoding="utf8") as file:
    for sample in new_samples:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")