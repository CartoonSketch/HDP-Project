from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import json
from tabpfn_client import TabPFNClassifier

# Import the training function from train_model.py
from train_model import train_model  # We'll modify train_model.py to have a function

# =========================
# 1️⃣ Train the model first
# =========================
model, MODEL_META = train_model()  # train_model() will return the trained model and metadata

# =========================
# 2️⃣ Start Flask app
# =========================
app = Flask(__name__)

FEATURES = MODEL_META['features']
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

os.makedirs("static/images/user", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get user inputs safely
        user_data = []
        for feature in FEATURES:
            value = request.form.get(feature)
            try:
                user_data.append(float(value))
            except (ValueError, TypeError):
                user_data.append(0.0)

        input_df = pd.DataFrame([user_data], columns=FEATURES)

        # Prediction
        prob = 0.0
        try:
            proba = model.predict_proba(input_df)
            if proba is not None and len(proba[0]) > 1:
                prob = proba[0][1] * 100
        except Exception as e:
            print("❌ Prediction error:", e)

        prediction = (
            "High Risk" if prob >= 60
            else "Medium Risk" if prob >= 30
            else "Low Risk"
        )

        # Save Pie Chart
        labels = ["No Risk", "Heart Disease Risk"]
        values = [100 - prob, prob]
        plt.figure(figsize=(5,5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, 
                colors=["#4CAF50","#E63946"])
        plt.title("Risk Probability")
        pie_path = "static/images/user/pie_chart.png"
        plt.savefig(pie_path)
        plt.close()

        # Save Bar Chart
        user_features = dict(zip(FEATURES, user_data))
        risky_features = {f: v for f, v in user_features.items() if v > 0}
        if risky_features:
            plt.figure(figsize=(8,5))
            plt.bar(risky_features.keys(), risky_features.values(), color="orange")
            plt.title("User Health Factors (Non-zero values)")
            plt.xticks(rotation=45, ha="right")
            bar_path = "static/images/user/bar_chart.png"
            plt.savefig(bar_path)
            plt.close()
        else:
            bar_path = None

        # Collect Suggestions
        feedback = [SUGGESTIONS[f] for f in risky_features if f in SUGGESTIONS]

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(prob, 2),
            inputs=user_features,
            pie_chart=pie_path,
            bar_chart=bar_path,
            feedback=feedback,
            model_accuracy=MODEL_META.get("accuracy")
        )

    return render_template("predict.html", features=FEATURES)

if __name__ == "__main__":
    app.run(debug=True)
