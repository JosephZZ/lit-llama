import json

# Load the first file for analysis
# eval_json = 'out/DPO/eval/gpt-4eval-aligner(blue)-adapter(red).json'
# eval_json = "out/DPO/eval/gpt-4eval-aligner(blue)-lora8(red).json"
# eval_json = "out/DPO/eval/gpt-4eval-aligner10-adapter.json"
# eval_json = "out/DPO/eval/gpt-4eval-aligner(blue)-lora8(red).json"
eval_json = "out/DPO/eval/gpt-3.5-turbo7B-aligner-Beta0.5-vs-Beta0.1-epoch3"
with open(eval_json, 'r') as file:
    data = json.load(file)


from collections import defaultdict

# Initializing structures to store data
scores_by_category = defaultdict(lambda: defaultdict(list))
win_loss_draw_by_category = defaultdict(lambda: {'win': 0, 'loss': 0, 'draw': 0})

# Processing data to calculate average scores and win-loss-draw statistics
for entry in data:
    category = entry['category']
    score1 = entry['score1']
    score2 = entry['score2']

    # Storing scores for average calculation
    scores_by_category[category]['red'].append(score1)
    scores_by_category[category]['blue'].append(score2)

    # Determining win, loss, or draw for each entry
    if score1 > score2:
        win_loss_draw_by_category[category]['win'] += 1
    elif score1 < score2:
        win_loss_draw_by_category[category]['loss'] += 1
    else:
        win_loss_draw_by_category[category]['draw'] += 1

# Calculating average scores
average_scores = {category: {model: sum(scores) / len(scores) for model, scores in models.items()}
                  for category, models in scores_by_category.items()}

print("average_scores, win_loss_draw_by_category", average_scores, win_loss_draw_by_category)



# Calculating total summary of scores and win-loss-draw for each model in both files

def calculate_total_summary(scores_by_category, win_loss_draw_by_category):
    total_scores = {'red': 0, 'blue': 0}
    total_wld = {'win': 0, 'loss': 0, 'draw': 0}
    total_entries = 0

    for category in scores_by_category:
        for model in total_scores:
            total_scores[model] += sum(scores_by_category[category][model])
            total_entries += len(scores_by_category[category][model])
        for result in total_wld:
            total_wld[result] += win_loss_draw_by_category[category][result]

    average_scores = {model: score / total_entries for model, score in total_scores.items()}
    return average_scores, total_wld

# Calculating for both files
total_summary_1 = calculate_total_summary(scores_by_category, win_loss_draw_by_category)

print("total_summary", total_summary_1)