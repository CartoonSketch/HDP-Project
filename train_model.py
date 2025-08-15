import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import pickle

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv('data/heart_data.csv')

# -----------------------------
# 2. Optional: Sample dataset for TabPFN efficiency
# -----------------------------
# TabPFN works best on small/medium datasets (<20k-30k rows)
sample_size = 30000  # adjust if needed
df_sample = df.sample(n=sample_size, random_state=42)

# -----------------------------
# 3. Define features and target
# -----------------------------
X = df_sample.drop('HeartDiseaseorAttack', axis=1)
y = df_sample['HeartDiseaseorAttack']

# -----------------------------
# 4. Split into train/test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Initialize TabPFN
# -----------------------------
# device='cpu' for CPU usage, can use 'cuda' if GPU available
model = TabPFNClassifier(device='cpu', n_ensemble_configs=32)

# -----------------------------
# 6. Train the model
# -----------------------------
print("Training TabPFN model... This may take several minutes.")
model.fit(X_train.values, y_train.values)

# -----------------------------
# 7. Evaluate accuracy
# -----------------------------
accuracy = model.score(X_test.values, y_test.values)
print(f"Model Accuracy on sampled test data: {accuracy:.2f}")

# -----------------------------
# 8. Save trained model
# -----------------------------
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Trained TabPFN model saved as 'heart_disease_model.pkl'")
