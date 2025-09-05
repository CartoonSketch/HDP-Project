import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Flask App
app = Flask(__name__)

# Load trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Function to generate Matplotlib figure and return base64 string
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64


# Health recommendations based on input
def generate_recommendations(inputs):
    feedback = []

    if inputs.get("HighBP", 0) == 1:
        feedback.append("Maintain healthy blood pressure by reducing salt intake and exercising regularly.")

    if inputs.get("HighChol", 0) == 1:
        feedback.append("Follow a heart-healthy diet to lower cholesterol levels.")

    if inputs.get("Smoker", 0) == 1:
        feedback.append("Quitting smoking will significantly reduce your heart disease risk.")

    if inputs.get("BMI", 0) > 30:
        feedback.append("Consider a weight management plan to lower your BMI.")

    if inputs.get("PhysActivity", 0) == 0:
        feedback.append("Engage in at least 30 minutes of physical activity daily.")

    if inputs.get("GenHlth", 0) >= 4:
        feedback.append("Schedule regular health checkups to improve your general health.")

    if not feedback:
        feedback.append("You are maintaining a healthy lifestyle. Keep it up!")

    return feedback


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        inputs = {
            "HighBP": int(request.form['HighBP']),
            "HighChol": int(request.form['HighChol']),
            "CholCheck": int(request.form['CholCheck']),
            "BMI": float(request.form['BMI']),
            "Smoker": int(request.form['Smoker']),
            "Stroke": int(request.form['Stroke']),
            "Diabetes": int(request.form['Diabetes']),
            "PhysActivity": int(request.form['PhysActivity']),
            "Fruits": int(request.form['Fruits']),
            "Veggies": int(request.form['Veggies']),
            "HvyAlcoholConsump": int(request.form['HvyAlcoholConsump']),
            "AnyHealthcare": int(request.form['AnyHealthcare']),
            "NoDocbcCost": int(request.form['NoDocbcCost']),
            "GenHlth": int(request.form['GenHlth']),
            "MentHlth": int(request.form['MentHlth']),
            "PhysHlth": int(request.form['PhysHlth']),
            "DiffWalk": int(request.form['DiffWalk']),
            "Sex": int(request.form['Sex']),
            "Age": int(request.form['Age']),
            "Education": int(request.form['Education']),
            "Income": int(request.form['Income'])
        }

        input_df = pd.DataFrame([inputs])

        # Prediction
        prediction = model.predict(input_df.values)[0]
        probability = model.predict_proba(input_df.values)[0][1] * 100

        risk_status = "High Risk" if prediction == 1 else "Low Risk"

        # Generate charts
        # Pie chart
        fig1, ax1 = plt.subplots()
        ax1.pie([probability, 100 - probability], labels=["Risk", "Safe"], autopct='%1.1f%%', colors=["#ff4d4d", "#4dff88"])
        pie_chart = plot_to_img(fig1)

        # Bar chart for risk factors
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(inputs.keys()), y=list(inputs.values()), ax=ax2)
        plt.xticks(rotation=90)
        plt.title("Patient Feature Profile")
        bar_chart = plot_to_img(fig2)

        # Generate feedback
        feedback = generate_recommendations(inputs)

        return render_template(
            'result.html',
            risk_status=risk_status,
            probability=round(probability, 2),
            inputs=inputs,
            pie_chart=pie_chart,
            bar_chart=bar_chart,
            feedback=feedback
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
