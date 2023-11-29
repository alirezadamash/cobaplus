import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ast

# Load the csv files
df1 = pd.read_csv('result/MNIST/resnet/Adam_Existing_231125172257.csv')
df2 = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_FR_231125193417.csv')

# Create a directory to save the plots
if not os.path.exists('Figures/confusion_matrix'):
    os.makedirs('Figures/confusion_matrix')

# Plot the heatmap of test_confusion_matrix for each epoch
for i in range(len(df1)):
    confusion_matrix1 = np.array(ast.literal_eval(df1['test_confusion_matrix'][i])).reshape(10, 10)
    sns.heatmap(confusion_matrix1, annot=True, fmt='d')
    plt.savefig(f'confusion_matrix/Adam_Existing_231125172257_epoch_{i}.png')
    plt.close()

for i in range(len(df2)):
    confusion_matrix2 = np.array(ast.literal_eval(df2['test_confusion_matrix'][i])).reshape(10, 10)
    sns.heatmap(confusion_matrix2, annot=True, fmt='d')
    plt.savefig(f'confusion_matrix/CoBAMSGrad2_FR_231125193417_epoch_{i}.png')
    plt.close()

