import kagglehub
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print("Path:", path)

normal_df = pd.read_csv(os.path.join(path, "ptbdb_normal.csv")).iloc[:, :-1]
anomaly_df = pd.read_csv(os.path.join(path, "ptbdb_abnormal.csv")).iloc[:, :-1]

normal_df["label"] = 0
anomaly_df["label"] = 1

data = pd.concat([normal_df, anomaly_df], axis=0).reset_index(drop=True)

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(data.drop("label", axis=1))
y = data["label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_all = np.vstack([X_train, X_test])
X_all_tsne = tsne.fit_transform(X_all)

X_train_tsne = X_all_tsne[:len(X_train)]
X_test_tsne = X_all_tsne[len(X_train):]

plt.figure(figsize=(10,6))
plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=y_train, cmap='coolwarm', alpha=0.3, s=20, label='Train')
plt.scatter(X_test_tsne[y_test==y_pred,0], X_test_tsne[y_test==y_pred,1], c='green', s=30, alpha=0.6, label='Test Correct')
plt.scatter(X_test_tsne[y_test!=y_pred,0], X_test_tsne[y_test!=y_pred,1], c='red', s=50, label='Test Incorrect')

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Gaussian Naive Bayes - Test Data projected with t-SNE")
plt.legend()
plt.show()
