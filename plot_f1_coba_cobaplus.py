import pandas as pd
import matplotlib.pyplot as plt

# Load the data
cobaplus_data = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_LS_231201224440.csv')
coba_data = pd.read_csv('result_coba/MNIST/resnet/CoBAMSGrad2_LS_231129210613.csv')

# Extract the f1 score every 100 epochs
cobaplus_f1 = cobaplus_data['test_f1']
coba_f1 = coba_data['test_f1']

# Create a new figure
plt.figure()

# Plot the f1 scores
plt.plot(cobaplus_f1, label='cobaplus_FR')
plt.plot(coba_f1, label='coba_FR')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score by Epoch')
plt.legend()

# Show the plot
plt.show()
