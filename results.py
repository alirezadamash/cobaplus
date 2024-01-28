import os
import pandas as pd

directory = 'result/MNIST/resnet'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        
        test_columns = [col for col in df.columns if 'test_' in col and 'confusion_matrix' not in col and 'test_loss' not in col]
        max_values = df[test_columns].idxmax()
        
        print("File: ", filename)
        for col, max_idx in max_values.items():
            max_val = df.loc[max_idx, col]
            epoch = df.loc[max_idx, 'epoch']
            print("Max value in {} is {:.4f} at epoch {}".format(col, max_val, epoch))
