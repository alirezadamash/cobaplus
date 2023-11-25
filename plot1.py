from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

color_dict = dict(
    HS='C0',
    FR='C1',
    PRP='C2',
    DY='C3',
    HZ='C4',
    Existing='C5',
)
marker_dict = dict(
    Existing='',
    HS='',
    FR='',
    PRP='',
    DY='',
    HZ='',
)

x_label_dict = dict(
    epoch='epoch',
    time='elapsed time [s]',
)


def get_linestyle(name: str, lr_type: str) -> str:
    if lr_type == 'Existing':
        return 'dotted'
    elif 'CM' in name:
        return 'solid'
    elif 'CoB' in name:
        return 'dashed'
    else:
        return 'dashed'

# get the list of csv files
csv_files = glob.glob("result/IMDb/lstm/*.csv")

# loop over the csv files
for csv_file in csv_files:

    # read the csv file
    data = pd.read_csv(csv_file)

    # get the optimizer name
    name = data["optimizer"].unique()[0]

    # get the test accuracy for the optimizer
    y_true = data["test_accuracy"]

    # create a binary label for the test accuracy (1 if above 0.5, 0 otherwise)
    y_true = (y_true > 0.5).astype(int)

    # create a score column for the test accuracy (same as y_true in this case)
    y_score = y_true.copy()

    # calculate the F1 score for the optimizer
    f1 = f1_score(y_true, y_score)
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
   
    # calculate the confusion matrix and check its shape before unpacking
    cm = confusion_matrix(y_true, y_score)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print("Unexpected shape for confusion matrix:", cm.shape)
        continue

    sensitivity = tp / (tp+fn)

    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.jpg')

    # plot the F1 score, precision, recall, sensitivity for the optimizer
    base_name, lr_type = name.split("_")
    color = color_dict[lr_type]
    linestyle = get_linestyle(base_name, lr_type)
    # plot the F1 score as a horizontal line
    plt.figure(figsize=(12, 9))
    plt.hlines(f1, 0, 100, label=f"{name} (F1 = {f1:.4f})", color=color, linestyle=linestyle,
               marker=marker_dict[lr_type])
    plt.hlines(precision, 0, 100, label=f"{name} (Precision = {precision:.4f})", color=color, linestyle=linestyle,
               marker=marker_dict[lr_type])
    plt.hlines(recall, 0, 100, label=f"{name} (Recall = {recall:.4f})", color=color, linestyle=linestyle,
               marker=marker_dict[lr_type])
    plt.hlines(sensitivity, 0, 100, label=f"{name} (Sensitivity = {sensitivity:.4f})", color=color, linestyle=linestyle,
               marker=marker_dict[lr_type])
    # add some labels and a legend
    plt.xlabel(x_label_dict["epoch"])
    plt.ylabel("Metrics")
    plt.title("Evaluation Metrics for MNIST Classification with perceptron2 Model")
    plt.legend()

    # save the figure
    plt.savefig("evaluation_metrics.jpg", dpi=300, bbox_inches="tight", pad_inches=.05)

    # close the figure
    plt.close()

