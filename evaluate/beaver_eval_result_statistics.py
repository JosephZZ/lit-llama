import json

result_file = "/home/zhouziheng/Desktop/lit-llama/out/beaverEvalComparisons/gpt-4-1106-previewalignerBlue_vs_3shotRed.json"

with open(result_file, encoding='utf-8') as f:
    results = json.load(f)

categories = set(problem['category'] for problem in results)
for category in categories:
    score1_list = [problem['score1'] for problem in results if problem['category'] == category]
    score2_list = [problem['score2'] for problem in results if problem['category'] == category]
    score1_average = sum(score1_list) / len(score1_list)
    score2_average = sum(score2_list) / len(score2_list)
    print(f'{category}: {score1_average :.2f} vs {score2_average:.2f}')
# compare category level win-loss-draw rate
for category in categories:
    win = sum(1 for problem in results if problem['category'] == category and problem['winner'] == 'red_model')
    loss = sum(1 for problem in results if problem['category'] == category and problem['winner'] == 'blue_model')
    draw = sum(1 for problem in results if problem['category'] == category and problem['winner'] == 'tie')
    print(f'{category}: {win} vs {loss} vs {draw}')
#print total average score
score1 = sum(problem['score1'] for problem in results)
score2 = sum(problem['score2'] for problem in results)
print(f'total: {score1 / len(results):.3f} vs {score2 / len(results):.3f}')
#print total win-loss-draw rate
win = sum(1 for problem in results if problem['winner'] == 'red_model')
loss = sum(1 for problem in results if problem['winner'] == 'blue_model')
draw = sum(1 for problem in results if problem['winner'] == 'tie')
print(f'total: {win} vs {loss} vs {draw}')
