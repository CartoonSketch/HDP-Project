import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabpfn_client import TabPFNClassifier

# Config paths
DATA_PATH = "data/HEART_DISEASE_PREDICTION_DATASET.csv"
TARGET = "HeartDiseaseorAttack"
MODEL_META_PATH = "model/heart_disease_model_meta.json"
USER_PLOTS_DIR = "static/images/user"
ANALYSIS_PLOTS_DIR = "static/images/analysis"
MAX_ROWS = 10000 

os.makedirs("model", exist_ok=True)
os.makedirs(USER_PLOTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_PLOTS_DIR, exist_ok=True)

# Load Dataset
df = pd.read_csv(DATA_PATH)

# Divide the dataset for TabPFN limit
if len(df) > MAX_ROWS:
    print(f"⚠️ Dataset has {len(df)} rows. So dividing into {MAX_ROWS} rows for TabPFN...")
    df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

X = df.drop(TARGET, axis=1)
y = df[TARGET]

FEATURES = list(X.columns)

# Train splited dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train TabPFN Model via API
model = TabPFNClassifier()
print("🤖 Training model using TabPFN API...")
model.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = (y_pred == y_test).mean()
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save dataset
MODEL_META = {"features": FEATURES, "accuracy": float(accuracy)}
with open(MODEL_META_PATH, "w") as f:
    json.dump(MODEL_META, f, indent=4)
print(f"💾 Trained Dataset saved at {MODEL_META_PATH}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease","Disease"],
            yticklabels=["No Disease","Disease"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(f"{ANALYSIS_PLOTS_DIR}/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(f"{ANALYSIS_PLOTS_DIR}/roc_curve.png")
plt.close()

# PCA Scatter
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(7,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", alpha=0.7)
plt.title("PCA of Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(f"{ANALYSIS_PLOTS_DIR}/pca_scatter.png")
plt.close()

# Density Plots
for col in X.columns:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=col, hue=TARGET, fill=True, common_norm=False, palette="Set1", alpha=0.6)
    plt.title(f"Density Plot - {col}")
    plt.savefig(f"{ANALYSIS_PLOTS_DIR}/density_{col}.png")
    plt.close()

print("📊 All Graph/plots analysis saved!")

# Flask App
app = Flask(__name__)

SUGGESTIONS = {
    "HighBP": "Consider regular exercise and a low-sodium diet to manage blood pressure.",
    "HighChol": "Limit fatty foods and increase fiber intake to lower cholesterol.",
    "BMI": "Maintain a healthy weight through balanced diet and physical activity.",
    "Smoker": "Quitting smoking greatly reduces heart disease risk.",
    "Stroke": "Consult a doctor for stroke management and prevention.",
    "Diabetes": "Manage sugar intake and monitor glucose levels regularly.",
    "PhysActivity": "Engage in at least 30 minutes of physical activity daily.",
    "Fruits": "Eat more fruits rich in vitamins and antioxidants.",
    "Veggies": "Include leafy vegetables in your meals daily.",
    "HvyAlcoholConsump": "Reduce alcohol consumption to protect heart health.",
    "AnyHealthcare": "Regular medical checkups are important for prevention.",
    "NoDocbcCost": "Seek affordable healthcare options to maintain health monitoring.",
    "GenHlth": "Work on improving general health through lifestyle changes.",
    "PhysHlth": "Pay attention to physical health; consult doctor if persistent issues.",
    "DiffWalk": "Physical therapy or regular walking may improve mobility.",
    "Sex": "Some risks vary by gender—consult doctor for personalized advice.",
    "Age": "With age, regular health checkups become more important."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method=="POST":
        user_data = []
        for feature in FEATURES:
            val = request.form.get(feature)
            try: user_data.append(float(val))
            except: user_data.append(0.0)

        input_df = pd.DataFrame([user_data], columns=FEATURES)

        prob = 0.0
        try:
            proba = model.predict_proba(input_df)
            prob = proba[0][1]*100
        except Exception as e:
            print("❌ Prediction error:", e)

        prediction = "High Risk" if prob>=60 else "Medium Risk" if prob>=30 else "Low Risk"

        # Pie Chart
        labels = ["No Risk","Heart Disease Risk"]
        values = [100-prob, prob]
        plt.figure(figsize=(5,5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#4CAF50","#E63946"])
        plt.title("Risk Probability")
        pie_path = f"{USER_PLOTS_DIR}/pie_chart.png"
        plt.savefig(pie_path)
        plt.close()

        # Bar Chart
        user_features = dict(zip(FEATURES, user_data))
        risky_features = {f:v for f,v in user_features.items() if v>0}
        if risky_features:
            plt.figure(figsize=(8,5))
            plt.bar(risky_features.keys(), risky_features.values(), color="orange")
            plt.title("User Health Factors (Non-zero)")
            plt.xticks(rotation=45, ha="right")
            bar_path = f"{USER_PLOTS_DIR}/bar_chart.png"
            plt.savefig(bar_path)
            plt.close()
        else:
            bar_path = None

        feedback = [SUGGESTIONS[f] for f in risky_features if f in SUGGESTIONS]

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(prob,2),
            inputs=user_features,
            pie_chart=pie_path,
            bar_chart=bar_path,
            feedback=feedback,
            model_accuracy=accuracy
        )

    return render_template("predict.html", features=FEATURES)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
