import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNClassifier
from sklearn.metrics import accuracy_score

# ==========================
# Paths
# ==========================
DATA_PATH = "data/HEART_DISEASE_PREDICTION_DATASET.csv" 
MODEL_META_PATH = "model/heart_disease_model_meta.json"

# ==========================
# Features & Target
# ==========================
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
]
TARGET = "HeartDiseaseorAttack"  

# ==========================
# Load dataset
# ==========================
df = pd.read_csv(DATA_PATH)

# Ensure target exists
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

# Split features and target
X = df[FEATURES]
y = df[TARGET]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# Train TabPFNClassifier
# ==========================
model = TabPFNClassifier()
model.fit(X_train.values, y_train.values, N_ensemble_configurations=1)  # N_ensemble_configurations=1 for speed

# Predict on test set
y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test.values, y_pred)

# ==========================
# Generate metadata
# ==========================
meta = {
    "features": FEATURES,
    "accuracy": round(float(accuracy), 4)
}

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save to JSON
with open(MODEL_META_PATH, "w") as f:
    json.dump(meta, f, indent=4)

print(f"✅ heart_disease_model_meta.json created with accuracy: {accuracy:.4f}")
