import matplotlib
matplotlib.use("Agg")        # Prevent GUI freezing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.cluster import KMeans

# ===========================
# 1) LOAD DATA
# ===========================
df = pd.read_csv(
    r"C:\Users\aakan\OneDrive\Desktop\ASSIGNMENT AK\7172\On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2023_1.csv",
    low_memory=False
)

# *** SPEED FIX — Reduce dataset size ***
df = df.sample(50000, random_state=42)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("\t", "")

print(df.columns.tolist())

# ===========================
# 2) SELECT FEATURES
# ===========================
selected_cols = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Operating_Airline", "Flight_Number_Operating_Airline",
    "Origin", "Dest", "Distance",
    "DepDelay", "ArrDelay", "DepDel15", "ArrDel15",
    "TaxiOut", "TaxiIn"
]

df = df[selected_cols].copy()
df = df.dropna()

df["Delay_Status"] = (df["ArrDelay"] >= 15).astype(int)

print(df.head())
print(df.shape)

# ===========================
# 3) GRAPHS (Saved)
# ===========================

# Distribution of arrival delays
plt.figure(figsize=(6,4))
df["ArrDelay"].hist(bins=50)
plt.title("Distribution of Arrival Delay")
plt.xlabel("Delay (min)")
plt.ylabel("Flight Count")
plt.savefig("plot_01.png", dpi=300)
plt.close()

# Class balance
plt.figure(figsize=(4,3))
df["Delay_Status"].value_counts().plot(kind="bar")
plt.title("Class Balance (Delayed vs Not Delayed)")
plt.xticks([0,1], ["No Delay", "Delayed"], rotation=0)
plt.savefig("plot_02.png", dpi=300)
plt.close()

# Encode categorical for heatmap
cat_cols = ["Operating_Airline", "Origin", "Dest"]
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("plot_heatmap.png", dpi=300)
plt.close()

# ===========================
# 4) SPLIT DATA
# ===========================
X = df.drop("Delay_Status", axis=1)
y = df["Delay_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)

# ===========================
# 5) SCALE + PCA
# ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# PCA Scatter
plt.figure(figsize=(6,5))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train, s=3, alpha=0.5)
plt.title("PCA Scatter Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("plot_pca.png", dpi=300)
plt.close()

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=clusters, cmap="viridis", s=3)
plt.title("K-Means Clustering on PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("plot_kmeans.png", dpi=300)
plt.close()

# ===========================
# 6) MODELS (FAST)
# ===========================
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(max_depth=8, random_state=42)
tree_model.fit(X_train, y_train)

svm_model = LinearSVC()   # <<< FAST SVM
svm_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

models = [
    ("Logistic Regression", log_model, X_test_scaled),
    ("Decision Tree", tree_model, X_test),
    ("Linear SVM", svm_model, X_test_scaled),
    ("Random Forest", rf_model, X_test),
    ("KNN", knn_model, X_test_scaled)
]

results = []

# ===========================
# 7) EVALUATION
# ===========================
for name, model, Xt in models:
    print(f"\n==== {name} ====")
    y_pred = model.predict(Xt)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"cm_{name}.png", dpi=300)
    plt.close()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
print(results_df)

# ===========================
# 8) FEATURE IMPORTANCE
# ===========================
importances = rf_model.feature_importances_
indices = pd.Series(importances, index=X.columns).sort_values()

plt.figure(figsize=(8,6))
indices.plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.savefig("feature_importance.png", dpi=300)
plt.close()
