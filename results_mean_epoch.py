import os
import pandas as pd

directory = 'result/MNIST/resnet'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        
        # فیلتر کردن داده‌ها برای اپوک‌های 15 تا 100
        df_filtered = df[(df['epoch'] >= 15) & (df['epoch'] <= 100)]
        
        test_columns = [col for col in df_filtered.columns if 'test_' in col and 'confusion_matrix' not in col and 'test_loss' not in col]
        mean_values = df_filtered[test_columns].mean()
        
        print("File: ", filename)
        for col, mean_val in mean_values.items():
            print("Mean value in {}: {:.4f}".format(col, mean_val))