import matplotlib.pyplot as plt

# # Updated data with new names and removed Aligner 20
# aligners_updated = [
#     "Aligner 1\n(Param 5k)", 
#     "Aligner 10\n(Param 42k)", 
#     "Aligner 100\n(Param 410k)", 
#     "Aligner 300\n(Param 1.2M)", 
#     "LLaMA-Adapter\n(Param 1.2M)", 
#     "LoRA\n(Param 4.2M)"
# ]
# accuracy_updated = [70/1319, 136/1319, 215/1319, 346/1319, 332/1319, 469/1319]

# # Creating the bar plot 
# plt.figure(figsize=(10,6))
# plt.bar(aligners_updated, accuracy_updated, color='PaleTurquoise')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.title('GSM8K Accuracy of Different Methods with Parameters')
# plt.ylim(0, max(accuracy_updated) + 0.05)  # Adding some space above the highest bar
# plt.xticks(rotation=45)
# plt.grid(axis='y')

# # Show the updated plot
# plt.show()



### line plot
import numpy as np

# Updated data with parameters as x-values
parameters = [5e3, 42e3, 410e3, 1.2e6, 1.2e6, 4.2e6]  # Parameters in order of aligners
accuracy_line_plot = [70/1319, 136/1319, 215/1319, 346/1319, 332/1319, 469/1319]
aligners_line_plot = [
    "Aligner 1 (5k)", 
    "Aligner 10 (42k)", 
    "Aligner 100 \n(410k)", 
    "", 
    "Aligner 300 \nLLaMA-Adapter\n(1.2M)", 
    "LoRA (4.2M)"
]
# Adjusting the x-axis to spread out the labels and avoid overlap
# Since the first two parameters are close in value, we adjust their positions slightly for better visibility

# Slightly modifying the positions of the first two parameters
parameters_adjusted = np.array(parameters)
parameters_adjusted[0] -= 2e4  # Shifting the first parameter slightly left
parameters_adjusted[1] += 2e4  # Shifting the second parameter slightly right

# Creating the adjusted line plot
plt.figure(figsize=(12,6))
plt.plot(parameters_adjusted, accuracy_line_plot, marker='o', color='PaleTurquoise')
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title('GSM8K Accuracy of Different Methods with Parameters')
plt.xticks(parameters_adjusted, aligners_line_plot, rotation=45)
plt.grid()

# Show the adjusted line plot
plt.show()

# Creating the line plot
plt.figure(figsize=(12,6))
plt.plot(parameters, accuracy_line_plot, marker='o', color='PaleTurquoise')
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title('GSM8K Accuracy of Different Methods with Parameters')
plt.xticks(parameters, aligners_line_plot, rotation=45)
plt.grid()

# Show the updated line plot
plt.show()

