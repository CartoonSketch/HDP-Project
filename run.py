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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Config paths
DATA_PATH = "data/HEART_DISEASE_PREDICTION_DATASET.csv"
TARGET = "HeartDiseaseorAttack"
MODEL_META_PATH = "model/heart_disease_model_meta.json"
USER_PLOTS_DIR = "static/images/user"
ANALYSIS_PLOTS_DIR = "static/images/analysis"
MAX_ROWS = 10000

# Create directories
os.makedirs("model", exist_ok=True)
os.makedirs(USER_PLOTS_DIR, exist_ok=True)
for subdir in ["tabpfn", "randomforest", "tree"]:
    os.makedirs(f"{ANALYSIS_PLOTS_DIR}/{subdir}", exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

X = df.drop(TARGET, axis=1)
y = df[TARGET]
FEATURES = list(X.columns)

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Initialize models
models = {
    "TabPFN": TabPFNClassifier(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

model_results = {}
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train each model & generate plots
for name, model in models.items():
    print(f"🤖 Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except:
        y_prob = np.zeros(len(y_test))
    
    acc = (y_pred == y_test).mean()
    print(f"✅ {name} Accuracy: {acc:.2f}")
    
    # Paths for this model
    plot_dir = f"{ANALYSIS_PLOTS_DIR}/{name.lower()}"
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"])
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = f"{plot_dir}/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend(loc="lower right")
    roc_path = f"{plot_dir}/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    
    # PCA Scatter
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", alpha=0.7)
    plt.title(f"{name} PCA Scatter")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    pca_path = f"{plot_dir}/pca_scatter.png"
    plt.savefig(pca_path)
    plt.close()
    
    # Density Plots
    density_paths = {}
    for col in X.columns:
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x=col, hue=TARGET, fill=True, common_norm=False, palette="Set1", alpha=0.6)
        plt.title(f"{name} Density - {col}")
        density_path = f"{plot_dir}/density_{col}.png"
        plt.savefig(density_path)
        plt.close()
        density_paths[col] = density_path.replace("static/", "")
    
    # Save results for template 
    model_results[name] = {
        "accuracy": round(acc*100,2),
        "confusion_matrix": cm_path.replace("static/", ""),
        "roc_curve": roc_path.replace("static/", ""),
        "pca_scatter": pca_path.replace("static/", ""),
        "density": density_paths
    }


# Save Model Meta
MODEL_META = {**{f"{name}_accuracy": model_results[name]["accuracy"] for name in model_results}}
with open(MODEL_META_PATH, "w") as f:
    json.dump(MODEL_META, f, indent=4)
print("💾 Saved model meta.")


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

        # Predictions for all models
        results = {}
        for name, model in models.items():
            try:
                prob = model.predict_proba(input_df)[0][1]*100
            except:
                prob = model.predict(input_df)[0]*100
            results[name] = {
                "prob": round(prob,2),
                "accuracy": model_results[name]["accuracy"],
                "confusion_matrix": model_results[name]["confusion_matrix"],
                "roc_curve": model_results[name]["roc_curve"],
                "pca_scatter": model_results[name]["pca_scatter"],
                "density": model_results[name]["density"]
            }

        # Overall prediction using TabPFN
        prob = results["TabPFN"]["prob"]
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
            pie_chart=pie_path.replace("static/", ""),
            bar_chart=bar_path.replace("static/", "") if bar_path else None,
            feedback=feedback,
            model_results=results
        )

    return render_template("predict.html", features=FEATURES)

if __name__=="__main__":
    app.run(debug=True)
