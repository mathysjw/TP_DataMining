import kagglehub
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print("Path:", path)

normal_df = pd.read_csv(os.path.join(path, "ptbdb_normal.csv")).iloc[:, :-1]
anomaly_df = pd.read_csv(os.path.join(path, "ptbdb_abnormal.csv")).iloc[:, :-1]

normal_df["label"] = 0
anomaly_df["label"] = 1

data = pd.concat([normal_df, anomaly_df], axis=0).reset_index(drop=True)

# NaN 

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(data.drop("label", axis=1))
y = data["label"].values

# Standardiser
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


tree = DecisionTreeClassifier(
    criterion="gini",      
    max_depth=8,           
    min_samples_split=10,
    random_state=42
)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)


print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Display graphs 

def plot_decision_tree_samples(X_test, y_test, y_pred, n_samples=5):

    idxs = np.random.choice(len(X_test), n_samples, replace=False)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(idxs):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(X_test[idx], label=f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.legend()
        plt.title(f"ECG sample {idx}")

    plt.tight_layout()
    plt.show()

plot_decision_tree_samples(X_test, y_test, y_pred)


plt.figure(figsize=(20, 12))
plot_tree(tree, filled=True, fontsize=6)
plt.title("Decision Tree")
plt.show()

