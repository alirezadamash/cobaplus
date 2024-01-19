import os
import pandas as pd

directory = 'result/MNIST/resnet'

results = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        
        test_columns = [col for col in df.columns if 'test_' in col and 'confusion_matrix' not in col and 'test_loss' not in col]
        mean_values = df[test_columns].mean()
        
        optimizer_name = filename.split('_')[0]
        results[optimizer_name] = mean_values

print("\\begin{table}[ht]")
print("\\begin{latin}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Optimizer} & \\textbf{Acc} & \\textbf{Pr} & \\textbf{Re} & \\textbf{F1} & \\textbf{MCC} & \\textbf{AUC-ROC} & \\textbf{AUPR} \\\\ \\hline")

for optimizer, values in results.items():
    print("\\textit{{\\textbf{{{}}}}} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \\hline".format(optimizer, values['test_accuracy'], values['test_precision'], values['test_recall'], values['test_f1'], values['test_mcc'], values['test_roc_auc'], values['test_aupr']))

print("\\end{tabular}")
print("\\end{latin}")
print("\\caption{نتایج معیارهای ارزیابی بهینه‌سازهای روش‌های پیشنهادی و موجود}")
print("\\end{table}")
