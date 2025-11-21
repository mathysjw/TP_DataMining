import kagglehub
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os


path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print("Path to dataset files:", path)

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


normal_df = pd.read_csv(os.path.join(path, "ptbdb_normal.csv")).iloc[:, :-1]
anomaly_df = pd.read_csv(os.path.join(path, "ptbdb_abnormal.csv")).iloc[:, :-1]

normal_df['label'] = 0
anomaly_df['label'] = 1

data = pd.concat([normal_df, anomaly_df], axis=0).reset_index(drop=True)

print("Nombre de NaN par colonne :")
print(data.isna().sum())

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop('label', axis=1))
y = data['label'].values


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


def plot_knn_samples(X_test, y_test, y_pred, n_samples=5):
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(X_test[idx], label=f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.legend()
        plt.title(f"ECG Sample {idx}")
    
    plt.tight_layout()
    plt.show()


plot_knn_samples(X_test, y_test, y_pred, n_samples=5)
