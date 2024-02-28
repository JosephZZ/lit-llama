import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
# Assuming 'previous_file_embeddings' is an array of shape [N, 4096] from your previous file
# And 'current_file_embeddings' is an array of shape [M, 4096] from your current file
# N and M are the number of embeddings in the previous and current files, respectively.
aligner1_path = "out/aligner/lit-llama-2-alpaca512/7B/1vector-start_layer0-lr0.009bs64-wu1.5/epoch-5.0.pth"
aligner10_path = "out/aligner/lit-llama-2-alpaca512/7B/10vector-start_layer2-lr0.009bs64-wu1.5/epoch-5.0.pth"
aligner20_path = "out/aligner/redo/lit-llama-2-alpaca512/7B/20vector-start_layer2-lr0.009bs64-wu2/epoch-8.0-valloss0.9019.pth"
aligner1_v2_path = "out/aligner/redo/lit-llama-2-alpaca512/7B/1vector-start_layer2-lr0.009bs64weightDecay0.02wu2/epoch-8.0-valloss0.9565"
aligner1_math_path = "out/aligner/redo/lit-llama-2-metaMath/7B/1vector-start_layer2-lr0.009bs64-wu1/epoch-3.0-valloss0.7474.pth"
aligner10_math_path = "out/aligner/redo/lit-llama-2-metaMath/7B/10vector-start_layer2-lr0.009bs64-wu1/epoch-3.0-valloss0.7005.pth"
aligner100_math_path = "out/aligner/redo/lit-llama-2-metaMath/7B/100vector-start_layer2-lr0.009bs64-wu0.3/epoch-3.0-valloss0.6941.pth"
aligner1_DPO_path = "out/DPO/aligner/lit-llama-2-beaver_safe2/7B/base_beaver_safe_alpacaStyle_SFT-1vector-start_layer2-lr0.0001bs32/epoch-3.0-iter-059999.pth"
aligner_checkpoint = torch.load(aligner1_math_path)
aligner_embedding = aligner_checkpoint['global_value_embedding.weight'].to(torch.float32).cpu().numpy()

## adapter_checkpoint = torch.load("/home/ziheng/ssd-drive1/projects/llm/lit-llama/out/adapter/redo/lit-llama-2-alpaca512/7B/lr0.009bs64.0wd0.02wu2/epoch-8.0-valloss0.8872.pth")
# adapter_checkpoint = torch.load("out/adapter/lit-llama-2-metaMath/7B/lr0.009bs64.0wd0.02wu1/epoch-3.0-valloss0.6179.pth")
# ade_keys = [k for k in adapter_checkpoint.keys() if 'weight' in k]
# adel = np.array([adapter_checkpoint[k].to(torch.float32).cpu().numpy() for k in ade_keys])
# adapter_embedding = adel.reshape(-1, adel.shape[-1])

adapter_checkpoint = torch.load(aligner100_math_path)
adapter_embedding = adapter_checkpoint['global_value_embedding.weight'].to(torch.float32).cpu().numpy()

# Combine the embeddings from both files into a single dataset
combined_embeddings = np.concatenate([aligner_embedding, adapter_embedding], axis=0)

# Apply t-SNE to reduce the dimensionality
tsne = TSNE(n_components=2, random_state=42) #, perplexity=10)  # n_components can be 2 or 3
reduced_embeddings = tsne.fit_transform(combined_embeddings)

# Splitting the reduced embeddings back into two groups for visualization
reduced_aligner = reduced_embeddings[:len(aligner_embedding)]
reduced_adapter = reduced_embeddings[len(adapter_embedding):]




plt.figure(figsize=(12, 8))  # Adjust figure size as needed
# Points from the current file (the rest)
# colors = np.random.rand(30, 3)
# colors[:, 0] = colors[:, 0] * 0.3  # Reduce the red component

# color_palette = plt.get_cmap('cool', 30)

# for i in range(1, 31):
#     color = color_palette(i-1)
#     start_idx = len(reduced_adapter) + (i-1) * 10
#     end_idx = start_idx + 10
#     plt.scatter(reduced_embeddings[start_idx:end_idx, 0], reduced_embeddings[start_idx:end_idx, 1], 
#                 c=color, label=f'Adapter Layer {i} Embeddings')  # Use the same color for all points in the group



# plt.figure(figsize=(9, 6))  # Adjust figure size as needed
# colors = np.random.rand(10, 3)
# colors[:, 0] = colors[:, 0] * 0.3  # Reduce the red component
# color_palette = plt.get_cmap('cool', 10)

# for position in range(10):  # Iterate through each position
#     color = color_palette(position)
#     for layer in range(30):  # Iterate through each layer
#         # Calculate the index of the current point
#         idx = len(reduced_adapter) + layer * 10 + position
#         # Plot the point
#         plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
#                     c=color, label=f'Adapter Position {position+1}' if layer == 0 else '')  


# Visualization

plt.scatter(reduced_embeddings[len(reduced_aligner):, 0], reduced_embeddings[len(reduced_aligner):, 1], 
            c='red', label='Aligner 100 Embeddings')
plt.scatter(reduced_embeddings[:len(reduced_aligner), 0], reduced_embeddings[:len(reduced_aligner), 1], 
            c='black', label='Aligner 1 Embeddings')

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')
plt.title('t-SNE visualization of all embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.subplots_adjust(right=0.7)

plt.savefig('tsne_plot.png', dpi=300)  # Adjust filename and dpi as needed

plt.show()
