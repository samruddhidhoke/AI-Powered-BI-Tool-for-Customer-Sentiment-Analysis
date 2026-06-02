# =====================================================
# PERFORMANCE METRICS GRAPH SCRIPT
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("Loading evaluation predictions...")

#input csv is output of train_issue_classifier
df = pd.read_csv("evaluation_predictions.csv")

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted"
)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(8,5))

bars = plt.bar(metrics.keys(), metrics.values())

plt.ylim(0, 1)
plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Score")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        f"{yval:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()


#saving model performance comparison as an png image file
print("Performance graph saved as model_performance_comparison.png")