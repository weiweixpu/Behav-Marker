import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')  # Or 'TkAgg', 'Qt5Agg', etc., based on your environment.

# Set the default font to serif (similar to Times New Roman)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Arial']
rcParams['font.size'] = 12

csv_files = [
r"E:\Parkinson's Project\data\results\plus 1007 data\first time\merged\alldata_densenet121_merged_test_predictions.csv",
r"E:\Parkinson's Project\data\results\plus 1007 data\first time\merged\alldata_resnet50_merged_test_predictions.csv",
r"E:\Parkinson's Project\data\results\plus 1007 data\first time\merged\alldata_vit_base_test_predictions.csv",
r"E:\Parkinson's Project\data\results\plus 1007 data\first time\merged\alldata_vit_large_test_predictions.csv",
r"E:\Parkinson's Project\data\results\plus 1007 data\first time\merged\alldata_foundation_test_predictions.csv",
]

# Corresponding labels for the 5 selected models
labels = [
    'DenseNet121',
    'ResNet50',
    'ViT-Base',
    'ViT-Large',
    'Foundation Model',
]

# Create a larger figure with higher DPI for better quality
plt.figure(figsize=(10, 8), dpi=300)

# Use a color-blind friendly palette for 5 colors
colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, 5))

n_bootstraps = 1000  # Number of bootstrap iterations
random_state = 42  # Random seed
alpha = 0.95  # Confidence interval significance level

for idx, csv_file in enumerate(csv_files):
    # Read CSV file
    data = pd.read_csv(csv_file)
    labels_col = data['label']
    predicted_probs = data['Predicted Probability']

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels_col, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Perform bootstrapping for confidence interval
    bootstrapped_auc = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(labels_col)), random_state=random_state + i)
        if len(np.unique(labels_col.iloc[indices])) < 2:
            continue
        fpr_boot, tpr_boot, _ = roc_curve(labels_col.iloc[indices], predicted_probs.iloc[indices])
        roc_auc_boot = auc(fpr_boot, tpr_boot)
        bootstrapped_auc.append(roc_auc_boot)

    # Calculate confidence interval
    sorted_auc_scores = np.array(bootstrapped_auc)
    sorted_auc_scores.sort()
    lower_bound = sorted_auc_scores[int((1.0 - alpha) / 2 * len(sorted_auc_scores))]
    upper_bound = sorted_auc_scores[int((1.0 + alpha) / 2 * len(sorted_auc_scores))]

    # If AUC is less than 0.5, reverse the curve and adjust the confidence interval
    if roc_auc < 0.5:
        fpr, tpr = 1 - fpr, 1 - tpr
        roc_auc = auc(fpr, tpr)
        lower_bound, upper_bound = 1 - upper_bound, 1 - lower_bound

    # Plot ROC curve with AUC and 95% CI
    plt.plot(fpr, tpr, color=colors[idx], lw=2,
             label=f'{labels[idx]} (AUC = {roc_auc:.3f}, 95% CI = [{lower_bound:.3f}, {upper_bound:.3f}])')

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)

# Set title and labels
plt.title('ROC Curves of Different Models of Multimodal Fusion', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)

# Customize the legend
plt.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

# Customize the grid
plt.grid(True, linestyle='--', alpha=0.7)

# Improve tick labels
plt.tick_params(axis='both', which='major', labelsize=12)

# Set the aspect of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Adjust layout and save the figure
plt.tight_layout()

# Save the figure with high quality settings
plt.savefig(r'E:\Parkinsons Project\Results\M1.pdf', format='pdf', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
