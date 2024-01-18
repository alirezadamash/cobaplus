import pandas as pd
import matplotlib.pyplot as plt
import os

# Get all csv files in the directory
csv_files = [f for f in os.listdir('result/MNIST/resnet/') if f.endswith('.csv') and 'const' not in f]

# Create a new figure with a defined figure size for better clarity
plt.figure(figsize=(10, 6))

# Define a dictionary to map full names to short names
name_mapping = {
    'Adam_Existing': 'Adam',
    'AdaGrad_Existing': 'AdaGrad',
    'AMSGrad_Existing': 'AMSGrad',
    'RMSProp_Existing': 'RMSProp',
    'CoBAMSGrad2_FR': 'Proposed_FR',
    'CoBAMSGrad2_CD': 'Proposed_CD',
    'CoBAMSGrad2_HS': 'Proposed_HS',
    'CoBAMSGrad2_LS': 'Proposed_LS',
    'CoBAMSGrad2_PRP': 'Proposed_PRP',
    'CoBAMSGrad2_DY': 'Proposed_DY'
}

# Loop through all csv files and plot the AUC ROC
for csv_file in csv_files:
    data = pd.read_csv(f'result/MNIST/resnet/{csv_file}')
    AUC_ROC = data['test_roc_auc']
    # Get the short name from the dictionary using the csv file name
    short_name = name_mapping.get('_'.join(csv_file.split('_')[:-1]), csv_file.split('.')[0])
    plt.plot(AUC_ROC, label=short_name)

# Add labels and title with larger font sizes
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('AUC/ROC', fontsize=14)
plt.title('AUC/ROC by Epoch', fontsize=16)
plt.legend(fontsize=12)

# Show the plot with a grid for better clarity
plt.grid(True)
plt.show()
