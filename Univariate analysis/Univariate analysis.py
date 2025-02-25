import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Arial']
rcParams['font.size'] = 12
# import matplotlib
# matplotlib.use('Agg')  # Or 'TkAgg', 'Qt5Agg', etc., based on your environment.

# Read data
train = pd.read_csv(r"E:\Parkinson's disease topic\Clinical\Clinical data\Data set division\Third time\train.csv", encoding='ISO-8859-1')

# Convert data types
train['label'] = train['label'].astype('category')
features = ['Distance', 'Mean speed', 'Line crossings', 'Center entries',
            'Center time', 'Center distance', 'Around entries', 'Around time', 'Around distance']

# Check which features actually exist in the data
existing_features = [f for f in features if f in train.columns]
print("Existing features:", ", ".join(existing_features))

# Convert existing features to numeric type
for feature in existing_features:
    train[feature] = pd.to_numeric(train[feature], errors='coerce')

plt.figure(figsize=(10, 8), dpi=300)
colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(existing_features)))

n_bootstraps = 1000 # Set the number of bootstrap samples
random_state = 42 # Random seed
alpha = 0.95 # Significance level of confidence interval

for idx, feature in enumerate(existing_features):
    fpr, tpr, _ = roc_curve(train['label'], train[feature])
    roc_auc = auc(fpr, tpr)

    # Bootstrap for confidence interval
    bootstrapped_auc = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(train)), random_state=random_state + i)
        if len(np.unique(train['label'].iloc[indices])) < 2:
            continue
        fpr_boot, tpr_boot, _ = roc_curve(train['label'].iloc[indices], train[feature].iloc[indices])
        roc_auc_boot = auc(fpr_boot, tpr_boot)
        bootstrapped_auc.append(roc_auc_boot)

    # Calculate confidence intervals
    sorted_auc_scores = np.array(bootstrapped_auc)
    sorted_auc_scores.sort()
    lower_bound = sorted_auc_scores[int((1.0 - alpha) / 2 * len(sorted_auc_scores))]
    upper_bound = sorted_auc_scores[int((1.0 + alpha) / 2 * len(sorted_auc_scores))]

    # Draw ROC curve
    plt.plot(fpr, tpr, color=colors[idx], lw=2,
             label=f'{feature} (AUC = {roc_auc:.3f}, 95% CI = [{lower_bound:.3f}, {upper_bound:.3f}])')

# Draw diagonal lines
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)

# Set title and labels
plt.title('ROC Curves of Different Features', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)

# Custom legend
plt.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

# Custom Grid
plt.grid(True, linestyle='--', alpha=0.7)

plt.tick_params(axis='both', which='major', labelsize=12)

# Set the aspect ratio of the graphic
plt.gca().set_aspect('equal', adjustable='box')

# Adjust the layout and save the graph
plt.tight_layout()
# plt.savefig('ROC_Curves_with_Features_and_CI.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Display the graph
plt.show()

# Print AUC value and confidence interval
for feature in existing_features:
    fpr, tpr, _ = roc_curve(train['label'], train[feature])
    roc_auc = auc(fpr, tpr)

    # Bootstrap for confidence interval
    bootstrapped_auc = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(train)), random_state=random_state + i)
        if len(np.unique(train['label'].iloc[indices])) < 2:
            continue
        fpr_boot, tpr_boot, _ = roc_curve(train['label'].iloc[indices], train[feature].iloc[indices])
        roc_auc_boot = auc(fpr_boot, tpr_boot)
        bootstrapped_auc.append(roc_auc_boot)

    sorted_auc_scores = np.array(bootstrapped_auc)
    sorted_auc_scores.sort()
    lower_bound = sorted_auc_scores[int((1.0 - alpha) / 2 * len(sorted_auc_scores))]
    upper_bound = sorted_auc_scores[int((1.0 + alpha) / 2 * len(sorted_auc_scores))]

    print(f"{feature}: AUC = {roc_auc:.3f}, 95% CI = [{lower_bound:.3f}, {upper_bound:.3f}]")


