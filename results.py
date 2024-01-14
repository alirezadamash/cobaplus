import os
import pandas as pd

directory = 'result/MNIST/'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        
        max_test_aupr_row = df[df['test_aupr'] == df['test_aupr'].max()]
        print("File: ", filename, "Epoch with max test_aupr: ", max_test_aupr_row['epoch'].values[0])