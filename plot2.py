from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import ast
import re

# read the result.csv file
data = pd.read_csv("result/MNIST/resnet/Adam_Existing_231125034832.csv")

# select the confusion matrix for Adam_Existing
confusion_matrix = data["test_confusion_matrix"][data["optimizer"] == "Adam_Existing"]

# convert the string into a list
def convert_to_list_of_lists(s):
    # Replace spaces between numbers with commas
    s = re.sub(r'(\d) +(\d)', r'\1, \2', s)
    s = re.sub(r'(\d) +', r'\1, ', s)
    s = re.sub(r' +(\d)', r', \1', s)
    return [list(map(int, row)) for row in ast.literal_eval(s)]

confusion_matrix_list = confusion_matrix.apply(convert_to_list_of_lists)

# convert the list into a numpy array
confusion_matrix_array = np.array(confusion_matrix_list.tolist())

# create a figure
plt.figure(figsize=(12, 9))

# plot the ROC AUC for Adam_Existing
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(confusion_matrix_array[:, i], confusion_matrix_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color_dict[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")

# save the figure
plt.savefig("roc_curve.jpg", dpi=300, bbox_inches="tight", pad_inches=.05)

# close the figure
plt.close()

x_label_dict = dict(
    epoch='epoch',
    time='elapsed time [s]',
)

# read the result.csv file
data = pd.read_csv("result/MNIST/resnet/Adam_Existing_231125034832.csv")

# select the confusion matrix for Adam_Existing
confusion_matrix = data["test_confusion_matrix"][data["optimizer"] == "Adam_Existing"]

# convert the string into a list
confusion_matrix_list = confusion_matrix.apply(convert_to_list_of_lists)

# convert the list into a numpy array
confusion_matrix_array = np.array(confusion_matrix_list.tolist())

# create a figure
plt.figure(figsize=(12, 9))

# plot the ROC AUC for Adam_Existing
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(confusion_matrix_array[:, i], confusion_matrix_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color_dict[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")

# save the figure
plt.savefig("roc_curve.jpg", dpi=300, bbox_inches="tight", pad_inches=.05)

# close the figure
plt.close()