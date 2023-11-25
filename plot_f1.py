import pandas as pd
import matplotlib.pyplot as plt

# Load the data
adam_data = pd.read_csv('result/MNIST/resnet/Adam_Existing_231125172257.csv')
cobams_data = pd.read_csv('result/MNIST/resnet/CoBAMSGrad2_FR_231125193417.csv')

# Extract the f1 score every 100 epochs
adam_f1 = adam_data['test_f1']
cobams_f1 = cobams_data['test_f1']

# Create a new figure
plt.figure()

# Plot the f1 scores
plt.plot(adam_f1, label='Adam')
plt.plot(cobams_f1, label='CoBAMSGrad2')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score by Epoch')
plt.legend()

# Show the plot
plt.show()
