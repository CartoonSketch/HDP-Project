# Heart Disease AI Predictor

The Heart Disease AI Predictor is a machine learning-based web application built using **Python, Flask, and scikit-learn**. It predicts the likelihood of a person having heart disease based on 20 health-related factors such as blood pressure, cholesterol, diabetes, lifestyle habits, and more.  

Users can provide their health information through a **user-friendly web form**, and the application will return a **risk prediction** along with the **probability percentage** and **model accuracy**.

## Features
- Predicts heart disease risk using 3 different models **TabPFN, RandomForest, DecisionTree**
- User-friendly web interface with **20 health-related questions**
- Displays prediction, probability (% chance), and model accuracy, suggestions and depth analysis
- Fully professional and clean design

## Workflow

1. Landing Page → Click “Let's Predict”

2. Input Form → Answer 20 health-related questions

3. Result Page → View risk prediction, probability, and model accuracy with full analysis

4. Predict again using the back button at the bottom

## Dataset Used

- The dataset contains 2 lakh+ Rows
- The dataset contains 20 factors related to heart health
- HeartDiseaseorAttack, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, PhysHlth, DiffWalk, Sex, Age
- All categorical/binary factors are converted to 0/1 for ML modeling.

- Numeric thresholds:
Age > 50 → 1, else 0
BMI > 30 → 1, else 0
PhysHlth > 10 days → 1, else 0

## Machine Learning (ML) Model Used

- TabPFN (Tabular Prior-data Fitted Network)
- Random Forest Classifier
- Tree Decision Classifier

- Accuracy: Approx. 98% (can vary depending on dataset size)

## Technologies Used

- Python
- Flask
- scikit-learn
- pandas
- numpy
- matplotlib
- pytorch
- HTML
- CSS
- Javascript
- JSON

## License

This project is licensed under the MIT License.

## Contributors

- Arnav Sharma (Team Lead)
- Akash Pandit (Team Co Lead)
- Sanjay Reddy (Team Member)
- Atharv Gupta (Team Member)
- Abhishek Mishra (Team Member)
- Shaurya Shukla (Team Member)
