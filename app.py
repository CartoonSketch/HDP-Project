from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load the trained TabPFN model
# -----------------------------
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# -----------------------------
# Landing page
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# Prediction form page
# -----------------------------
@app.route('/predict')
def predict():
    return render_template('predict.html')

# -----------------------------
# Handle prediction
# -----------------------------
@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract 19 features from the form
        features = [
            float(request.form.get('HighBP')),
            float(request.form.get('HighChol')),
            float(request.form.get('CholCheck')),
            float(request.form.get('BMI')),
            float(request.form.get('Smoker')),
            float(request.form.get('Stroke')),
            float(request.form.get('Diabetes')),
            float(request.form.get('PhysActivity')),
            float(request.form.get('Fruits')),
            float(request.form.get('Veggies')),
            float(request.form.get('HvyAlcoholConsump')),
            float(request.form.get('AnyHealthcare')),
            float(request.form.get('NoDocbcCost')),
            float(request.form.get('GenHlth')),
            float(request.form.get('PhysHlth')),
            float(request.form.get('DiffWalk')),
            float(request.form.get('Sex')),
            float(request.form.get('Age'))
        ]

        # Convert to numpy array for TabPFN
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)[0]  # 0 or 1
        probability = model.predict_proba(features_array)[0][1] * 100  # % chance of heart disease

        # Convert prediction to user-friendly text
        if prediction == 1:
            pred_text = "High Risk of Heart Disease"
        else:
            pred_text = "Low Risk of Heart Disease"

        # Optional: Display model accuracy (from training script)
        # You can hardcode the accuracy here or compute dynamically if saved
        model_accuracy = 0.85  # Example: 85%

        return render_template(
            'result.html',
            prediction=pred_text,
            probability=round(probability, 2),
            accuracy=round(model_accuracy * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
