from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# List of input features (must match your dataset order!)
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect user inputs from form
        user_inputs = {}
        values = []
        for feature in FEATURES:
            val = request.form.get(feature)
            if val is None or val == "":
                val = 0
            val = float(val)
            user_inputs[feature] = val
            values.append(val)

        values = np.array(values).reshape(1, -1)

        # Predict with TabPFN model
        prediction = model.predict(values)[0]
        probs = model.predict_proba(values)[0]
        confidence = round(float(max(probs)) * 100, 2)

        # Generate advice based on inputs
        advice = []
        if user_inputs["BMI"] > 30:
            advice.append("Your BMI indicates obesity. Consider weight management and regular exercise.")
        if user_inputs["Smoker"] == 1:
            advice.append("Smoking increases risk. Quitting smoking can significantly improve heart health.")
        if user_inputs["HighBP"] == 1:
            advice.append("You have high blood pressure. Reduce salt intake and manage stress.")
        if user_inputs["HighChol"] == 1:
            advice.append("High cholesterol detected. Focus on a balanced diet with less saturated fat.")
        if user_inputs["Diabetes"] == 1:
            advice.append("Diabetes increases heart risk. Monitor blood sugar and follow medical advice.")
        if user_inputs["PhysActivity"] == 0:
            advice.append("Increase physical activity. Aim for at least 30 minutes of daily exercise.")
        if not advice:
            advice.append("Your inputs suggest good lifestyle habits. Keep maintaining them for heart health!")

        # Simulated feature importance (TabPFN does not give real importance scores directly)
        feature_names = FEATURES
        feature_importance = [abs(val) / (sum(values[0]) + 1e-5) * 100 for val in values[0]]

        return render_template(
            'result.html',
            prediction=prediction,
            confidence=confidence,
            user_inputs=user_inputs,
            advice=advice,
            feature_names=feature_names,
            feature_importance=feature_importance
        )

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
