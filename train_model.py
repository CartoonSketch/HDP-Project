import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import pickle

# 1. Load the dataset
df = pd.read_csv('data/HEART_DISEASE_PREDICTION_DATASET.csv')

# 2. Dividing dataset size into 30k chunks for TabPFN efficiency
sample_size = 30000  
df_sample = df.sample(n=sample_size, random_state=42)

# 3. Define features and target
X = df_sample.drop('HeartDiseaseorAttack', axis=1)
y = df_sample['HeartDiseaseorAttack']

# 4. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize TabPFN using Nivida GTX/RTX GPU
model = TabPFNClassifier(device='cuda', n_ensemble_configs=32)

# 6. Training the model
print("Training TabPFN model... This may take several minutes.")
model.fit(X_train.values, y_train.values)

# 7. Evaluate accuracy
accuracy = model.score(X_test.values, y_test.values)
print(f"Model Accuracy on test data: {accuracy:.2f}")

# 8. Save trained model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Trained TabPFN model saved as 'heart_disease_model.pkl'")
