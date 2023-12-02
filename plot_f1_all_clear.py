import pandas as pd
import matplotlib.pyplot as plt
import os

# Get all csv files in the directory
csv_files = [f for f in os.listdir('result/MNIST/resnet/') if f.endswith('.csv')]

# Create a new figure with a defined figure size for better clarity
plt.figure(figsize=(10, 6))

# Loop through all csv files and plot the f1 scores
for csv_file in csv_files:
    data = pd.read_csv(f'result/MNIST/resnet/{csv_file}')
    f1_scores = data['test_f1']
    plt.plot(f1_scores, label=csv_file.split('.')[0])

# Add labels and title with larger font sizes
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('F1 Score by Epoch', fontsize=16)
plt.legend(fontsize=12)

# Show the plot with a grid for better clarity
plt.grid(True)
plt.show()
