import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from tabpfn import TabPFNClassifier  

app = Flask(__name__)


# Load Models
TABPFN_MODEL = TabPFNClassifier(device="cpu")

RF_MODEL_PATH = "model/random_forest.pkl"
DT_MODEL_PATH = "model/decision_tree.pkl"

rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
dt_model = joblib.load(DT_MODEL_PATH) if os.path.exists(DT_MODEL_PATH) else None

# Metadata if you want to use, for example:
MODEL_META = {
    "TabPFN": {"accuracy": "98.01%"},
    "RandomForest": {"accuracy": "98.10%"}, 
    "DecisionTree": {"accuracy": "92.33%"}   
}



# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user inputs from form
        user_features = {
            "age": float(request.form.get("age", 0)),
            "cholesterol": float(request.form.get("cholesterol", 0)),
            "blood_pressure": float(request.form.get("blood_pressure", 0)),
            "max_heart_rate": float(request.form.get("max_heart_rate", 0)),
            "blood_sugar": float(request.form.get("blood_sugar", 0))
        }

        input_df = pd.DataFrame([user_features])

        results = {}

        # TabPFN
        try:
            proba_tab = TABPFN_MODEL.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            print("TabPFN error:", e)
            proba_tab = 0

        pred_tab = (
            "High Risk" if proba_tab >= 60 else
            "Medium Risk" if proba_tab >= 30 else
            "Low Risk"
        )

        results["TabPFN"] = {
            "prob": round(proba_tab, 2),
            "pred": pred_tab,
            "acc": MODEL_META["TabPFN"]["accuracy"]
        }

        # Random Forest 
        if rf_model:
            try:
                proba_rf = rf_model.predict_proba(input_df)[0][1] * 100
                pred_rf = (
                    "High Risk" if proba_rf >= 60 else
                    "Medium Risk" if proba_rf >= 30 else
                    "Low Risk"
                )
                results["RandomForest"] = {
                    "prob": round(proba_rf, 2),
                    "pred": pred_rf,
                    "acc": MODEL_META["RandomForest"]["accuracy"]
                }
            except Exception as e:
                print("RandomForest error:", e)
                results["RandomForest"] = {"prob": None, "pred": "Error", "acc": None}
        else:
            results["RandomForest"] = {"prob": None, "pred": "Model not trained", "acc": None}

        # Decision Tree 
        if dt_model:
            try:
                proba_dt = dt_model.predict_proba(input_df)[0][1] * 100
                pred_dt = (
                    "High Risk" if proba_dt >= 60 else
                    "Medium Risk" if proba_dt >= 30 else
                    "Low Risk"
                )
                results["DecisionTree"] = {
                    "prob": round(proba_dt, 2),
                    "pred": pred_dt,
                    "acc": MODEL_META["DecisionTree"]["accuracy"]
                }
            except Exception as e:
                print("DecisionTree error:", e)
                results["DecisionTree"] = {"prob": None, "pred": "Error", "acc": None}
        else:
            results["DecisionTree"] = {"prob": None, "pred": "Model not trained", "acc": None}

        # Result Page 
        return render_template(
            "result.html",
            results=results,
            inputs=user_features
        )


if __name__ == '__main__':
    app.run(debug=True)
