import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("Crop_recommendation.csv")

# Feature Engineering
df["avg_nutrients"] = df[["N", "P", "K"]].mean(axis=1)
df["N_to_K_ratio"] = df["N"] / (df["K"] + 1e-3)

X = df.drop("label", axis=1)
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split: 60 train / 20 val / 20 test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# --- Decision Tree ---
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
val_acc_dtc = accuracy_score(y_val, dtc.predict(X_val))
print("Decision Tree:", val_acc_dtc)

with open("Final stuff/decision_tree_summary.txt", "w") as f:
    f.write("Decision Tree Classifier Summary\n")
    f.write(f"Validation Accuracy: {val_acc_dtc:.4f}\n")
    f.write(f"Parameters: {dtc.get_params()}\n")

# --- Random Forest Baseline ---
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
val_acc_rfc = accuracy_score(y_val, rfc.predict(X_val))
print("Random Forest:", val_acc_rfc)

with open("Final stuff/rfc_summary.txt", "w") as f:
    f.write("Random Forest Classifier Summary\n")
    f.write(f"Validation Accuracy: {val_acc_rfc:.4f}\n")
    f.write(f"Parameters: {rfc.get_params()}\n")

# --- RFE + RFC ---
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe.fit(X_train, y_train)
selected = X_train.columns[rfe.support_]

rfc_rfe = RandomForestClassifier(random_state=42)
rfc_rfe.fit(X_train[selected], y_train)
val_acc_rfe = accuracy_score(y_val, rfc_rfe.predict(X_val[selected]))
print("RFE Random Forest:", val_acc_rfe)

with open("Final stuff/rfc_rfe_summary.txt", "w") as f:
    f.write("Random Forest with RFE Summary\n")
    f.write(f"Top 5 Features: {list(selected)}\n")
    f.write(f"Validation Accuracy: {val_acc_rfe:.4f}\n")
    f.write(f"Parameters: {rfc_rfe.get_params()}\n")

# --- Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

val_acc_tuned = accuracy_score(y_val, grid.predict(X_val))
print("Tuned RFC:", val_acc_tuned)

with open("Final stuff/rfc_tuned_summary.txt", "w") as f:
    f.write("Tuned Random Forest Summary\n")
    f.write(f"Validation Accuracy: {val_acc_tuned:.4f}\n")
    f.write(f"Best Parameters: {grid.best_params_}\n")
    f.write(f"Full Grid Search Results:\n{grid.cv_results_}\n")

# --- Final Test Accuracy ---
test_acc = accuracy_score(y_test, grid.predict(X_test))
print("Final Test Accuracy:", test_acc)

with open("Final stuff/final_test_evaluation.txt", "w") as f:
    f.write("Final Test Set Evaluation\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Classification Report:\n{classification_report(y_test, grid.predict(X_test))}\n")
