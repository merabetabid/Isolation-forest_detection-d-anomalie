# ================================
# 1. IMPORTS
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)

import warnings
warnings.filterwarnings("ignore")

# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv("creditcard.csv")

print("Shape:", df.shape)
print(df["Class"].value_counts())

# ================================
# 3. PREPROCESSING
# ================================
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 4. PCA
# ================================
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# ================================
# 5. MODELS
# ================================
contamination = 0.0017

# Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
iso.fit(X_train)
y_pred_iso = iso.predict(X_test)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.0017
)

y_pred_lof = lof.fit_predict(X_test)

y_pred_lof = np.where(y_pred_lof == 1, 0, 1)

print(confusion_matrix(y_test, y_pred_lof))
print(classification_report(y_test, y_pred_lof))

# SVM
svm = OneClassSVM(kernel='rbf', gamma='auto', nu=contamination)
svm.fit(X_train)
y_pred_svm = svm.predict(X_test)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))

# ================================
# 6. METRICS
# ================================
precision_iso = precision_score(y_test, y_pred_iso)
precision_lof = precision_score(y_test, y_pred_lof)
precision_svm = precision_score(y_test, y_pred_svm)

recall_iso = recall_score(y_test, y_pred_iso)
recall_lof = recall_score(y_test, y_pred_lof)
recall_svm = recall_score(y_test, y_pred_svm)

# ================================
# 7. HISTOGRAMME PRECISION
# ================================
models = ['IF', 'LOF', 'SVM']
precisions = [precision_iso, precision_lof, precision_svm]

plt.figure(figsize=(8,5))
plt.bar(models, precisions)
plt.title("Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision")
plt.show()

# ================================
# 8. HISTOGRAMME RECALL
# ================================
recalls = [recall_iso, recall_lof, recall_svm]

plt.figure(figsize=(8,5))
plt.bar(models, recalls)
plt.title("Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall")
plt.show()

# ================================
# 9. ISOLATION FOREST CONTAMINATION
# ================================
contaminations = [0.001, 0.005, 0.01]
if_preds = {}

for c in contaminations:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(X_train)
    pred = model.predict(X_test)
    pred = np.where(pred == 1, 0, 1)
    if_preds[c] = pred

# ================================
# 10. VISUALISATION PCA
# ================================

# Réalité
plt.figure(figsize=(6,5))
plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c=y_test, s=5)
plt.title("Real Data")
plt.show()

# Contamination plots
for c, pred in if_preds.items():
    plt.figure(figsize=(6,5))
    plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c=pred, s=5)
    plt.title(f"Isolation Forest (contamination={c})")
    plt.show()
   
   
    # ================================
# 11. HISTOGRAMMES OPTIMISATION IF
# ================================

# Ajouter le cas réel (0.0017)
if_preds[0.0017] = y_pred_iso

 #affichage
cont_labels = sorted(if_preds.keys())

precision_if = []
recall_if = []

for c in cont_labels:
    precision_if.append(precision_score(y_test, if_preds[c]))
    recall_if.append(recall_score(y_test, if_preds[c]))

# ================================
# HISTOGRAMME PRECISION IF
# ================================
plt.figure(figsize=(8,5))

labels = [str(c) for c in cont_labels]

plt.bar(labels, precision_if)

plt.title("Precision - Isolation Forest (Optimization)")
plt.xlabel("Contamination")
plt.ylabel("Precision")

plt.show()

# ================================
# HISTOGRAMME RECALL IF
# ================================
plt.figure(figsize=(8,5))

plt.bar(labels, recall_if)

plt.title("Recall - Isolation Forest (Optimization)")
plt.xlabel("Contamination")
plt.ylabel("Recall")

plt.show()
# ================================
# 12. MATRICES DE CONFUSION
# ================================

import seaborn as sns

# ================================
# MATRICES MODELS
# ================================
models_preds = {
    "Isolation Forest": y_pred_iso,
    "LOF": y_pred_lof,
    "SVM": y_pred_svm
}

plt.figure(figsize=(12,4))

i = 1
for name, pred in models_preds.items():
    plt.subplot(1,3,i)
   
    cm = confusion_matrix(y_test, pred)
   
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
   
    i += 1

plt.tight_layout()
plt.show()


# ================================
# MATRICES IF (CONTAMINATION)
# ================================

# ajouter le cas réel
if_preds[0.0017] = y_pred_iso

cont_labels = sorted(if_preds.keys())

plt.figure(figsize=(12,8))

i = 1
for c in cont_labels:
    plt.subplot(2,2,i)
   
    cm = confusion_matrix(y_test, if_preds[c])
   
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"IF (c={c})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
   
    i += 1

plt.tight_layout()
plt.show()