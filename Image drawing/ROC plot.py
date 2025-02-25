"""
@time : 2024/3/1 9:23
@auth : HuZhenyuan
@file : ROC plot.py
"""
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_combined_roc_curve(train_csv, test_csv, vali_csv):
    # Read CSV file
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    vali_df = pd.read_csv(vali_csv)

    fpr_train, tpr_train, _ = roc_curve(train_df['label'], train_df['Predicted Probability'])
    fpr_test, tpr_test, _ = roc_curve(test_df['label'], test_df['Predicted Probability'])
    fpr_vali, tpr_vali, _ = roc_curve(vali_df['label'], vali_df['Predicted Probability'])

    # Calculate the AUC for each set
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)
    auc_vali = auc(fpr_vali, tpr_vali)

    # Draw ROC
    plt.figure(figsize=(8, 6))

    plt.plot(fpr_train, tpr_train, color='blue', label='Train AUC = %0.4f' % auc_train)
    plt.plot(fpr_test, tpr_test, color='green', label='Test AUC = %0.4f' % auc_test)
    plt.plot(fpr_vali, tpr_vali, color='red', label='Validation AUC = %0.4f' % auc_vali)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.show()

if __name__ == '__main__':
    train_csv = r"E:\data\results\Foundation model\TM_train_predictions.csv"
    test_csv = r"E:\data\results\Foundation model\TM_test_predictions.csv"
    vali_csv = r"E:\data\results\Foundation model\TM_vali_predictions.csv"
    plot_combined_roc_curve(train_csv, test_csv, vali_csv)
