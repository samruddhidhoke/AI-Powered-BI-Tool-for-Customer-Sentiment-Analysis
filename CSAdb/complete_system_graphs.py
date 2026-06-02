# =====================================================
# COMPLETE SYSTEM GRAPH GENERATION
# Saves All Graphs as Images
# =====================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# 1️⃣ Create Output Folder
# -------------------------------------

output_folder = r"D:\CSA\system_performance_graphs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Saving graphs inside folder:", output_folder)

# -------------------------------------
# 2️⃣ Load Final Output
# -------------------------------------

df = pd.read_csv("final_ai_output.csv")

# -------------------------------------
# 3️⃣ Business Output Distribution
# -------------------------------------

plt.figure(figsize=(8,5))
df["BusinessOutput"].value_counts().plot(kind="bar")

plt.title("Business Output Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "business_output_distribution.png"), dpi=300)
plt.close()

# -------------------------------------
# 4️⃣ Issue Confidence Distribution
# -------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df["IssueConfidence"], bins=20)

plt.title("Issue Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "confidence_distribution.png"), dpi=300)
plt.close()

# -------------------------------------
# 5️⃣ Issue Category Distribution
# -------------------------------------

plt.figure(figsize=(8,5))
df["IssueType"].value_counts().plot(kind="bar")

plt.title("Issue Category Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "issue_category_distribution.png"), dpi=300)
plt.close()

# -------------------------------------
# 6️⃣ Sentiment Distribution
# -------------------------------------

plt.figure(figsize=(8,5))
df["SentimentBucket"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "sentiment_distribution.png"), dpi=300)
plt.close()

print("\nAll system performance graphs saved successfully.")