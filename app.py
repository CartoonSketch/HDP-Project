# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Optional: Hardcoded model accuracy from training
MODEL_ACCURACY = 0.85  # Replace with actual accuracy from train_model.py

# 1. Landing page
@app.route("/")
def index():
    return render_template("index.html")

# 2. Input form page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    return render_template("predict.html")

# 3. Result page
@app.route("/result", methods=["POST"])
def result():
    # Collect inputs from form
    try:
        features = [
            int(request.form["HighBP"]),
            int(request.form["HighChol"]),
            int(request.form["CholCheck"]),
            int(request.form["BMI"]),
            int(request.form["Smoker"]),
            int(request.form["Stroke"]),
            int(request.form["Diabetes"]),
            int(request.form["PhysActivity"]),
            int(request.form["Fruits"]),
            int(request.form["Veggies"]),
            int(request.form["HvyAlcoholConsump"]),
            int(request.form["AnyHealthcare"]),
            int(request.form["NoDocbcCost"]),
            int(request.form["GenHlth"]),
            int(request.form["PhysHlth"]),
            int(request.form["DiffWalk"]),
            int(request.form["Sex"]),
            int(request.form["Age"])
        ]

        # Convert to numpy array
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1] * 100  # % chance

        # Convert prediction to text
        if prediction == 1:
            prediction_text = "High risk of heart disease"
        else:
            prediction_text = "Low risk of heart disease"

        return render_template(
            "result.html",
            prediction=prediction_text,
            probability=round(probability, 2),
            accuracy=int(MODEL_ACCURACY * 100)
        )

    except Exception as e:
        return f"Error: {e}"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
