import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import ast
import numpy as np
import re

# Load the data
adam_data = pd.read_csv('result/MNIST/resnet/Adam_Existing_231125172257.csv')
cobams_data = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_FR_231125193417.csv')

# Extract the confusion matrix for each epoch
for cm in adam_data['test_confusion_matrix']:
    processed_cm = '[' + re.sub(r'(\d)\s*', r'\1, ', cm) + ']'
    print(processed_cm)
    try:
        ast.literal_eval(processed_cm)
    except ValueError as e:
        print(f"Error for {processed_cm}: {e}")

for cm in cobams_data['test_confusion_matrix']:
    processed_cm = '[' + re.sub(r'(\d)\s*', r'\1, ', cm) + ']'
    print(processed_cm)
    try:
        ast.literal_eval(processed_cm)
    except ValueError as e:
        print(f"Error for {processed_cm}: {e}")

print(adam_cm.shape)

# For each epoch, plot the confusion matrix
for epoch in range(len(adam_cm)):
    # Create a new figure
    plt.figure()

    # Plot the confusion matrix for Adam optimizer
    sns.heatmap(adam_cm[epoch], annot=True, fmt='d')
    plt.title('Adam Confusion Matrix for Epoch ' + str(epoch))
    plt.show()

    # Plot the confusion matrix for CoBAMSGrad2 optimizer
    sns.heatmap(cobams_cm[epoch], annot=True, fmt='d')
    plt.title('CoBAMSGrad2 Confusion Matrix for Epoch ' + str(epoch))
    plt.show()
