# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load dataset
data = pd.read_csv("data/heart_data.csv")

# 2. Inspect dataset (optional, for debugging)
print("Columns in dataset:", data.columns)
print("First 5 rows:\n", data.head())

# 3. Preprocessing
# Mapping categorical/binary features to 0/1 if needed

# Sex: Male=1, Female=0
if data['Sex'].dtype == 'object':
    data['Sex'] = data['Sex'].map({'Male': 1, 'Female': 0})

# GenHlth: Excellent/Good=0, Fair/Poor=1
if data['GenHlth'].dtype == 'object':
    data['GenHlth'] = data['GenHlth'].map({
        'Excellent': 0, 'Very good': 0, 'Good': 0,
        'Fair': 1, 'Poor': 1
    })

# Numeric thresholds
# Age > 50 -> 1, else 0
data['Age'] = data['Age'].apply(lambda x: 1 if x > 50 else 0)

# BMI > 30 -> 1, else 0
data['BMI'] = data['BMI'].apply(lambda x: 1 if x > 30 else 0)

# PhysHlth > 10 days -> 1, else 0
data['PhysHlth'] = data['PhysHlth'].apply(lambda x: 1 if x > 10 else 0)

# Ensure all other binary features are int (0/1)
binary_features = ['HighBP','HighChol','CholCheck','Smoker','Stroke','Diabetes',
                   'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare',
                   'NoDocbcCost','DiffWalk']
for feature in binary_features:
    data[feature] = data[feature].astype(int)

# 4. Define features and target
X = data.drop('HeartDiseaseorAttack', axis=1)
y = data['HeartDiseaseorAttack']

# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save trained model
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nTrained model saved as 'heart_disease_model.pkl'")
