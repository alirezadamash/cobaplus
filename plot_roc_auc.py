import pandas as pd
import matplotlib.pyplot as plt

# Load the data
adam_data = pd.read_csv('result/MNIST/resnet/Adam_Existing_231125172257.csv')
cobams_data = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_FR_231125193417.csv')

# Extract the roc_auc score
adam_roc_auc = adam_data['test_roc_auc']
cobams_roc_auc = cobams_data['test_roc_auc']

# Create a new figure
plt.figure()

# Plot the roc_auc scores
plt.plot(adam_roc_auc, label='Adam')
plt.plot(cobams_roc_auc, label='CoBAPlus_FR')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('ROC AUC Score')
plt.title('ROC AUC Score by Epoch')
plt.legend()

# Show the plot
plt.show()
