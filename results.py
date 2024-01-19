import os
import pandas as pd

directory = 'result/MNIST/resnet'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        
        test_columns = [col for col in df.columns if 'test_' in col and 'confusion_matrix' not in col and 'test_loss' not in col]
        max_values = df[test_columns].max()
        
        print("File: ", filename)
        for col, max_val in max_values.items():
            print("Max value in {}: {:.4f}".format(col, max_val))
