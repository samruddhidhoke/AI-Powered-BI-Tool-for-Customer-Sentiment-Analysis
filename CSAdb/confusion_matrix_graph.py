# =====================================================
# CONFUSION MATRIX GRAPH SCRIPT
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Loading evaluation predictions...")


#input csv is output of train_issue_classifier
df = pd.read_csv("evaluation_predictions.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

# Reconstruct label names properly
label_encoder = LabelEncoder()
label_encoder.fit(y_true.tolist() + y_pred.tolist())
class_names = label_encoder.classes_

print("Class Names:", class_names) 
cm = confusion_matrix(y_true, y_pred)

# Convert to percentage view
cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(9,7))

#generation of heatmap representation
sns.heatmap(
    cm_percentage,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Confusion Matrix (Percentage View)", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()


#final saving of generated confusion matrix image
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


print("Confusion matrix saved as confusion_matrix.png")