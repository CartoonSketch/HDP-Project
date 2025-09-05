import os
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNClassifier
from app import app  # Import Flask app

# ======================
# Dataset & Features
# ======================
DATA_PATH = "data/HEART_DISEASE_PREDICTION_DATASET.csv"  # <-- your CSV path
TARGET = "HeartDiseaseorAttack"

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
]

# ======================
# Load dataset
# ======================
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Train TabPFNClassifier
# ======================
model = TabPFNClassifier()
model.fit(X_train.values, y_train.values, overwrite_warning=True)

# ======================
# Evaluate accuracy
# ======================
accuracy = model.score(X_test.values, y_test.values)
print(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# ======================
# Save model
# ======================
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/heart_disease_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model saved at {MODEL_PATH}")

# ======================
# Generate metadata JSON
# ======================
meta = {
    "features": FEATURES,
    "target": TARGET,
    "accuracy": round(accuracy*100, 2)
}
META_PATH = "model/heart_disease_model_meta.json"
with open(META_PATH, "w") as f:
    json.dump(meta, f)
print(f"✅ Metadata saved at {META_PATH}")

# ======================
# Start Flask app
# ======================
if __name__ == "__main__":
    app.run(debug=True)
