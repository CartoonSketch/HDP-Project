import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn_client import TabPFNClassifier
import os, json
import joblib   

def train_model():
    # Load Dataset
    df = pd.read_csv("data/HEART_DISEASE_PREDICTION_DATASET.csv")
    X = df.drop("HeartDiseaseorAttack", axis=1)
    y = df["HeartDiseaseorAttack"]

    # Split dataset for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("model", exist_ok=True)

    # TabPFN (API) 
    tabpfn = TabPFNClassifier()
    print("Training TabPFN via API...")
    tabpfn.fit(X_train, y_train)

    y_pred_tab = tabpfn.predict(X_test)
    y_prob_tab = tabpfn.predict_proba(X_test)[:, 1]
    acc_tab = accuracy_score(y_test, y_pred_tab)
    print(f"✅ TabPFN Model Trained with Accuracy: {acc_tab:.2f}")

    # Save TabPFN metadata 
    MODEL_META_TAB = {"features": list(X.columns), "accuracy": float(acc_tab)}

    with open("model/tabpfn_meta.json", "w") as f:
        json.dump(MODEL_META_TAB, f, indent=4)

    # Random Forest 
    rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
    print("Training Random Forest Model...")
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"✅ Random Forest Model Trained with Accuracy: {acc_rf:.2f}")

    joblib.dump(rf, "model/random_forest.pkl")

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=None)
    print("Training Decision Tree Model...")
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print(f"✅ Decision Tree Model Trained with Accuracy: {acc_dt:.2f}")

    joblib.dump(dt, "model/decision_tree.pkl")

    # Save overall metadata
    META = {
        "TabPFN": float(acc_tab),
        "RandomForest": float(acc_rf),
        "DecisionTree": float(acc_dt),
        "features": list(X.columns)
    }
    with open("model/models_meta.json", "w") as f:
        json.dump(META, f, indent=4)

    print("💾 All models saved in /model")

    os.makedirs("static/images/analysis", exist_ok=True)

    # Confusion Matrix Comparison
    cm = confusion_matrix(y_test, y_pred_tab)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix (TabPFN)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("static/images/analysis/confusion_matrix_tabpfn.png")
    plt.close()

    # ROC Curves comparison
    fpr_tab, tpr_tab, _ = roc_curve(y_test, y_prob_tab)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

    plt.figure(figsize=(7,6))
    plt.plot(fpr_tab, tpr_tab, color="blue", label=f"TabPFN (AUC={auc(fpr_tab,tpr_tab):.2f})", lw=2)
    plt.plot(fpr_rf, tpr_rf, color="blue", label=f"Random Forest (AUC={auc(fpr_rf,tpr_rf):.2f})", lw=2)
    plt.plot(fpr_dt, tpr_dt, color="blue", label=f"Decision Tree (AUC={auc(fpr_dt,tpr_dt):.2f})", lw=2)
    plt.plot([0,1], [0,1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.savefig("static/images/analysis/roc_comparison.png")
    plt.close()

    print("📊 Comparison plots saved in static/images/analysis/")

    return {"tabpfn": acc_tab, "random_forest": acc_rf, "decision_tree": acc_dt}

if __name__ == "__main__":
    train_model()
