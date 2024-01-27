import pandas as pd
import matplotlib.pyplot as plt

# Load the data
cobaplus_data = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_LS_231201224440.csv')
coba_data = pd.read_csv('result/MNIST/resnet/Adam_Existing_231201210940.csv')

# Extract the f1 score every 100 epochs
cobaplus_f1 = cobaplus_data['test_roc_auc']
coba_f1 = coba_data['test_roc_auc']

# Create a new figure
plt.figure()

# Plot the f1 scores
plt.plot(cobaplus_f1, label='Proposed_CG_Optimizer_LS')
plt.plot(coba_f1, label='Adam')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('AUC/ROC Score')
plt.title('AUC/ROC Score by Epoch')
plt.legend()

# Show the plot
plt.show()