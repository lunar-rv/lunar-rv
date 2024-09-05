import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrices for LWR and LR models
small_lwr = np.array([[1931.0, 688.0], [58.0, 2561.0]])
medium_lwr = np.array([[2223.0, 396.0], [78.0, 2541.0]])
large_lwr = np.array([[2431.0, 188.0], [89.0, 2530.0]])

small_lr = np.array([[1733.0, 886.0], [39.0, 2580.0]])
medium_lr = np.array([[2042.0, 577.0], [66.0, 2553.0]])
large_lr = np.array([[2336.0, 283.0], [88.0, 2531.0]])

# Labels for the heatmaps
x_labels = ["Safe", "Anomaly"]
y_labels = ["Safe", "Anomaly"]

# Function to plot confusion matrix heatmap
def plot_confusion_matrix(matrix, title, ax):
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax, xticklabels=x_labels, yticklabels=y_labels)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# Create the figure and axes
fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns for LWR and LR small, medium, large

# Plot each confusion matrix in its respective subplot
plot_confusion_matrix(small_lwr, "LWR Small", axes[0, 0])
plot_confusion_matrix(small_lr, "LR Small", axes[0, 1])

plot_confusion_matrix(medium_lwr, "LWR Medium", axes[1, 0])
plot_confusion_matrix(medium_lr, "LR Medium", axes[1, 1])

plot_confusion_matrix(large_lwr, "LWR Large", axes[2, 0])
plot_confusion_matrix(large_lr, "LR Large", axes[2, 1])

# Adjust the layout
plt.tight_layout()
plt.show()
