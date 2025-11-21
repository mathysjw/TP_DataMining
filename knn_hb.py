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
from sklearn.decomposition import PCA
import os


path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print("Path :", path)

# Listing files
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


normal_df = pd.read_csv(os.path.join(path, "ptbdb_normal.csv")).iloc[:, :-1]
anomaly_df = pd.read_csv(os.path.join(path, "ptbdb_abnormal.csv")).iloc[:, :-1]

normal_df['label'] = 0 #ok
anomaly_df['label'] = 1 #anomaly

data = pd.concat([normal_df, anomaly_df], axis=0).reset_index(drop=True)

print("NaNs number by column")
print(data.isna().sum())

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop('label', axis=1))
y = data['label'].values


scaler = StandardScaler() # mean = 0, standard deviation = 1 
X = scaler.fit_transform(X)

# 80% of the data for the training, 20% for the tests

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test, y_pred)) # precision, recall, f1-score and support
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Display the graphs

def plot_knn_neighbors(knn, X_train, y_train, X_test, y_test, index=0, pca=None):
    
    if index < 0 or index >= len(X_test):
        raise ValueError(f"index must be in [0, {len(X_test)-1}]")

    if pca is None:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(np.vstack([X_train, X_test]))
    else:
        X_reduced = pca.transform(np.vstack([X_train, X_test]))

    X_train_2d = X_reduced[:len(X_train)]
    X_test_2d  = X_reduced[len(X_train):]

    distances, indices = knn.kneighbors([X_test[index]])
    distances = distances[0]
    indices = indices[0]

    print(f"\nTest={index}")
    print(f"True label = {y_test[index]}")
    print("k neighbor :", indices)
    print("Distances :", np.round(distances, 4))
    print("Neighbors labels :", y_train[indices])

    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_train_2d[:,0], X_train_2d[:,1],
        c=y_train, cmap='coolwarm', s=12, alpha=0.4, label='Train'
    )

    plt.scatter(
        X_test_2d[index,0], X_test_2d[index,1],
        c='green', s=200, edgecolor='black', label=f"Test index={index} ,(true={y_test[index]})", zorder = 5
    )


    plt.scatter(
        X_train_2d[indices,0], X_train_2d[indices,1],
        c='yellow', s=140, edgecolor='black', label=f"{knn.n_neighbors} voisins (train)"
    )

    plt.title("KNN — point test + k voisins (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

chosen_index = 10 if len(X_test) > 10 else 0
plot_knn_neighbors(knn, X_train, y_train, X_test, y_test, index=chosen_index)