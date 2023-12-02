import pandas as pd
import matplotlib.pyplot as plt
import os

# Get all csv files in the directory
csv_files = [f for f in os.listdir('result/MNIST/resnet/') if f.endswith('.csv')]

# Create a new figure
plt.figure()

# Loop through all csv files and plot the f1 scores
for csv_file in csv_files:
    data = pd.read_csv(f'result/MNIST/resnet/{csv_file}')
    f1_scores = data['test_f1']
    plt.plot(f1_scores, label=csv_file.split('.')[0])

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score by Epoch')
plt.legend()

# Show the plot
plt.show()
