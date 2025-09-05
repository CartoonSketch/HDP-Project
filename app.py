from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Features used in dataset
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
]

# Suggestions mapping
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

# Ensure user plot folder exists
os.makedirs("static/images/user", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get user inputs
        user_data = []
        for feature in FEATURES:
            value = request.form.get(feature)
            try:
                user_data.append(float(value))
            except:
                user_data.append(0.0)

        input_df = pd.DataFrame([user_data], columns=FEATURES)

        # Prediction
        prob = model.predict_proba(input_df.values)[0][1] * 100
        prediction = "High Risk" if prob >= 60 else "Medium Risk" if prob >= 30 else "Low Risk"

        # Save Pie Chart
        labels = ["No Risk", "Heart Disease Risk"]
        values = [100 - prob, prob]
        plt.figure(figsize=(5,5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#4CAF50","#E63946"])
        plt.title("Risk Probability")
        pie_path = "static/images/user/pie_chart.png"
        plt.savefig(pie_path)
        plt.close()

        # Save Bar Chart (showing risky inputs only)
        user_features = dict(zip(FEATURES, user_data))
        risky_features = {f: v for f,v in user_features.items() if v > 0}
        plt.figure(figsize=(8,5))
        plt.bar(risky_features.keys(), risky_features.values(), color="orange")
        plt.title("User Health Factors (Non-zero values)")
        plt.xticks(rotation=45, ha="right")
        bar_path = "static/images/user/bar_chart.png"
        plt.savefig(bar_path)
        plt.close()

        # Collect Suggestions
        feedback = []
        for f, v in risky_features.items():
            if f in SUGGESTIONS:
                feedback.append(SUGGESTIONS[f])

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(prob,2),
            inputs=user_features,
            pie_chart=pie_path,
            bar_chart=bar_path,
            feedback=feedback
        )

    return render_template("predict.html", features=FEATURES)

if __name__ == "__main__":
    app.run(debug=True)
