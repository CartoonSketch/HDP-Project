import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabpfn_client import TabPFNClassifier   
import os, json

def train_model():
    # ================================
    # Load Dataset
    # ================================
    df = pd.read_csv("data/HEART_DISEASE_PREDICTION_DATASET.csv")

    X = df.drop("HeartDiseaseorAttack", axis=1)
    y = df["HeartDiseaseorAttack"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ================================
    # Train TabPFN Model via API
    # ================================
    model = TabPFNClassifier()   # ✅ use classifier, not client

    print("🚀 Sending data to TabPFN remote API for training...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"✅ Model Accuracy on test data: {accuracy:.2f}")

    # ================================
    # Save metadata
    # ================================
    MODEL_META = {
        "features": list(X.columns),
        "accuracy": float(accuracy)
    }
    os.makedirs("model", exist_ok=True)
    with open("model/heart_disease_model_meta.json", "w") as f:
        json.dump(MODEL_META, f, indent=4)

    print("💾 Model metadata saved as 'model/heart_disease_model_meta.json'")
    print("   (Model is not pickled – app.py will instantiate TabPFNClassifier fresh and use API.)")

    # ================================
    # Create directory for plots
    # ================================
    os.makedirs("static/images/analysis", exist_ok=True)

    # ================================
    # Confusion Matrix
    # ================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("static/images/analysis/confusion_matrix.png")
    plt.close()

    # ================================
    # ROC Curve
    # ================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("static/images/analysis/roc_curve.png")
    plt.close()

    # ================================
    # PCA Visualization
    # ================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", alpha=0.7)
    plt.title("PCA of Features (2 Components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("static/images/analysis/pca_plot.png")
    plt.close()

    # ================================
    # Density Plots of Features
    # ================================
    for col in X.columns:
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x=col, hue="HeartDiseaseorAttack", fill=True, common_norm=False, palette="Set1", alpha=0.6)
        plt.title(f"Density Plot - {col}")
        plt.savefig(f"static/images/analysis/density_{col}.png")
        plt.close()

    # ================================
    # Scatter Plots of Features (vs Age)
    # ================================
    if "Age" in X.columns:
        for col in X.columns:
            if col != "Age":
                plt.figure(figsize=(6,4))
                sns.scatterplot(data=df, x="Age", y=col, hue="HeartDiseaseorAttack", palette="Set1", alpha=0.6)
                plt.title(f"Scatter Plot - Age vs {col}")
                plt.savefig(f"static/images/analysis/scatter_Age_{col}.png")
                plt.close()

    print("📊 All global analysis plots saved in static/images/analysis/")

    return model, MODEL_META

# Only run training if called directly (optional)
if __name__ == "__main__":
    train_model()
