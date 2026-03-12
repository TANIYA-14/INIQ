import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data, preprocess_data, split_and_save
from feature_engineering import feature_engineering

def evaluate_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_performance_raw.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'student_performance_model.pkl')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first.")
        return

    print("Loading model...")
    artifacts = joblib.load(MODEL_PATH)
    clf = artifacts['model']
    # Re-process data to get Test set (Ideally we should save test set, but for this simple pipe we re-generate)
    # Note: Consistency relies on random_state
    
    df = load_data(RAW_DATA_PATH)
    df = feature_engineering(df)
    X, y, le, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_save(X, y)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Classification Report
    target_names = [str(c) for c in le.classes_]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)
    
    with open(os.path.join(REPORTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    print("Confusion matrix saved.")
    
    # Feature Importance
    importances = clf.feature_importances_
    features = X.columns
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[features[i] for i in indices])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'))
    print("Feature importance plot saved.")

if __name__ == "__main__":
    evaluate_model()
