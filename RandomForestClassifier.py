import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ChatGPTDecisionTree import RandomForest
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Extract features and target variable
X = df.drop(columns=["label"])  # Feature columns
y = df["label"]  # Target variable

# Encode target labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize the RandomForest classifier
rf_classifier = RandomForest(n_estimators=100, max_depth = 4, min_samples_split=2)

# Train the model
rf_classifier.train(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)