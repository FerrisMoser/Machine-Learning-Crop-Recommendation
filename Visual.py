import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Load and prepare data ---
df = pd.read_csv("Crop_recommendation.csv")
df["avg_nutrients"] = df[["N", "P", "K"]].mean(axis=1)
df["N_to_K_ratio"] = df["N"] / (df["K"] + 1e-3)
X = df.drop("label", axis=1)
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Split: 60% train, 20% val, 20% test ---
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# --- Re-train Models JUST FOR VISUALS ---
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe_selector.fit(X_train, y_train)
rfe_selected_features = X_train.columns[rfe_selector.support_]

rfc_tuned = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    cv=3, n_jobs=-1, verbose=1
)
rfc_tuned.fit(X_train, y_train)

# --- Visuals ---

# 🎄 Decision Tree Plot
plt.figure(figsize=(20, 10))
plot_tree(dtc, feature_names=X.columns, class_names=le.classes_, filled=True, max_depth=3)
plt.title("Decision Tree (Max Depth = 3)")
plt.tight_layout()
plt.savefig("decision_tree_visualization.png")
plt.close()

# 🔥 Feature Importances from Baseline RFC
plt.figure(figsize=(10, 6))
sns.barplot(x=baseline_model.feature_importances_, y=X.columns)
plt.title("Feature Importances - Baseline RFC")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.close()

# 🧠 RFE Feature Ranking
rfe_ranking = rfe_selector.ranking_
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=rfe_ranking)
plt.title("RFE Feature Ranking (Lower = Better)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rfe_feature_ranking.png")
plt.close()

# 🌈 PCA Crop Distribution
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
df["pca1"], df["pca2"] = proj[:, 0], proj[:, 1]
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="label", palette="tab20", s=60, edgecolor="k")
plt.title("PCA Crop Distribution")
plt.tight_layout()
plt.savefig("pca_crop_projection.png")
plt.close()

# ✅ Confusion Matrix from tuned RFC
final_preds = rfc_tuned.best_estimator_.predict(X_test)
conf = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(12, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Greens')
plt.title("Final Test Confusion Matrix (Tuned RFC)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_final_test.png")
plt.close()

# ⚖️ Agreement Matrix: Baseline vs Tuned
baseline_preds = baseline_model.predict(X_test)
tuned_preds = rfc_tuned.best_estimator_.predict(X_test)
agree = pd.crosstab(baseline_preds, tuned_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(agree, annot=True, fmt="d", cmap="Blues")
plt.title("Agreement Matrix: Baseline vs Tuned RFC")
plt.tight_layout()
plt.savefig("rfc_agreement_matrix.png")
plt.close()

changed = np.sum(baseline_preds != tuned_preds)
print(f"🔁 Predictions changed between Baseline and Tuned RFC: {changed} / {len(y_test)}")

# 📊 Validation vs Test Accuracy (more meaningful)
acc_df = pd.DataFrame({
    "Split": ["Validation", "Test"],
    "Accuracy": [
        accuracy_score(y_val, baseline_model.predict(X_val)),
        accuracy_score(y_test, baseline_model.predict(X_test)),
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=acc_df, x="Split", y="Accuracy")
plt.title("Validation vs Test Accuracy - Baseline RFC")
plt.ylim(0.95, 1.01)
plt.tight_layout()
plt.savefig("val_test_accuracy_comparison.png")
plt.close()

# 📉 Learning Curve Plot (Validation vs Test Accuracy)
train_sizes, train_scores, val_scores = learning_curve(baseline_model, X_trainval, y_trainval, cv=3, train_sizes=np.linspace(0.1, 1.0, 10))
val_means = np.mean(val_scores, axis=1)
test_means = [accuracy_score(y_test, baseline_model.predict(X_test))] * len(train_sizes)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, val_means, label="Validation Accuracy")
plt.plot(train_sizes, test_means, label="Test Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve: Validation vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve_val_test.png")
plt.close()

# 🧩 Confusion Matrix on Validation Set
val_conf = confusion_matrix(y_val, baseline_model.predict(X_val))
plt.figure(figsize=(10, 6))
sns.heatmap(val_conf, annot=True, fmt='d', cmap='Oranges')
plt.title("Validation Set Confusion Matrix - Baseline RFC")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("val_confusion_matrix_baseline.png")
plt.close()

# 📦 Feature Importance from Tuned Model
plt.figure(figsize=(10, 6))
sns.barplot(x=rfc_tuned.best_estimator_.feature_importances_, y=X.columns)
plt.title("Feature Importances - Tuned RFC")
plt.tight_layout()
plt.savefig("feature_importance_tuned.png")
plt.close()
