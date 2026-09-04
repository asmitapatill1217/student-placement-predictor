import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "student_placement_dataset.csv")
MODEL_DIR = os.path.join(BASE, "model")
GRAPH_DIR = os.path.join(BASE, "static", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

sns.set_style("whitegrid")

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "tenth_percent", "twelfth_percent", "cgpa", "internships",
    "projects", "backlogs", "communication_skill", "extra_curricular",
]

X = df[FEATURES]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
print(classification_report(y_test, pred))

joblib.dump(model, os.path.join(MODEL_DIR, "placement_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
    f.write(f"{acc*100:.2f}")

# ---------- Graphs used across the dashboards ----------

# 1. Placement distribution (pie)
plt.figure(figsize=(5, 5))
df["placed"].value_counts().rename({0: "Not Placed", 1: "Placed"}).plot.pie(
    autopct="%1.1f%%", colors=["#ff6b6b", "#4dd08c"], ylabel=""
)
plt.title("Overall Placement Distribution")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "placement_pie.png"), dpi=120)
plt.close()

# 2. CGPA vs Placement (box plot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="placed", y="cgpa", data=df, palette=["#ff6b6b", "#4dd08c"])
plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.title("CGPA vs Placement Status")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "cgpa_box.png"), dpi=120)
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(7, 5))
corr = df[FEATURES + ["placed"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

# 4. Internships vs placement rate (bar)
plt.figure(figsize=(6, 4))
rate = df.groupby("internships")["placed"].mean() * 100
rate.plot(kind="bar", color="#5b8def")
plt.ylabel("Placement Rate (%)")
plt.title("Placement Rate by Internship Count")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "internship_bar.png"), dpi=120)
plt.close()

# 5. Package distribution (histogram, placed only)
plt.figure(figsize=(6, 4))
sns.histplot(df[df["placed"] == 1]["package_lpa"], bins=20, kde=True, color="#4dd08c")
plt.title("Package Distribution (Placed Students)")
plt.xlabel("Package (LPA)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "package_hist.png"), dpi=120)
plt.close()

print("Model + graphs saved.")
