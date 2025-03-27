import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------
# Load and Prepare Data
# --------------------
df = pd.read_csv("Crop_recommendation.csv")

# Dataset summary (inline doc)
# ~2200 entries, 22 crop types
# Features: N, P, K, temperature, humidity, rainfall, pH
# Clean, balanced, no missing values

# --------------------
# Feature Engineering
# --------------------
df["avg_nutrients"] = df[["N", "P", "K"]].mean(axis=1)
df["N_to_K_ratio"] = df["N"] / (df["K"] + 1)

X = df.drop("label", axis=1)
y = df["label"]

# --------------------
# Train/Validation/Test Split
# --------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

# --------------------
# Baseline Decision Tree Classifier
# --------------------
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_val_dtc = dtc.predict(X_val)
print("\n[Decision Tree Validation Accuracy]", accuracy_score(y_val, y_val_dtc))
print(classification_report(y_val, y_val_dtc))

# Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtc, feature_names=X.columns, class_names=dtc.classes_, filled=True, max_depth=3)
plt.title("Sample Decision Tree Visualization (Max Depth 3)")
plt.tight_layout()
plt.savefig("decision_tree_visualization.png")
plt.show()

# --------------------
# Random Forest - Baseline
# --------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

print("\n[Random Forest Validation Accuracy]", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Feature Importance Plot
importances = model.feature_importances_
feat_names = X.columns
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importances (Baseline Model)")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.show()

# --------------------
# Feature Selection with RFE (using RFC as estimator)
# --------------------
rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe_selector.fit(X_train, y_train)
selected_features = X_train.columns[rfe_selector.support_]
print("\n[Top 5 Selected Features by RFE]", list(selected_features))

# Re-train with selected features only
X_train_rfe = X_train[selected_features]
X_val_rfe = X_val[selected_features]

rfe_model = RandomForestClassifier(random_state=42)
rfe_model.fit(X_train_rfe, y_train)
y_val_rfe = rfe_model.predict(X_val_rfe)
print("[RFE Random Forest Validation Accuracy]", accuracy_score(y_val, y_val_rfe))
print(classification_report(y_val, y_val_rfe))

# --------------------
# GridSearchCV - Hyperparameter Tuning
# --------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n[Best Parameters Found]", grid_search.best_params_)

# Evaluate tuned model on validation
tuned_val_pred = grid_search.best_estimator_.predict(X_val)
print("[Tuned Validation Accuracy]", accuracy_score(y_val, tuned_val_pred))
print(classification_report(y_val, tuned_val_pred))

# Final Evaluation on Test Set
tuned_test_pred = grid_search.best_estimator_.predict(X_test)
print("\n[Final Test Accuracy]", accuracy_score(y_test, tuned_test_pred))
print(classification_report(y_test, tuned_test_pred))

plt.figure(figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, tuned_test_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (Final Test Evaluation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_final_test.png")
plt.show()
