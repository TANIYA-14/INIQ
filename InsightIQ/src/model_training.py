import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data, preprocess_data, split_and_save
from feature_engineering import feature_engineering

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_performance_raw.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'student_performance_model.pkl')
    
    print("Loading data...")
    try:
        df = load_data(RAW_DATA_PATH)
    except FileNotFoundError:
        print("Raw data not found, generating it now...")
        # Fallback or error
        raise

    print("Applying Feature Engineering...")
    df = feature_engineering(df)
    
    print("Preprocessing...")
    X, y, le, scaler = preprocess_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_and_save(X, y)
    
    print("Training Random Forest Classifier...")
    # Baseline RFC
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on Train
    train_pred = clf.predict(X_train)
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    
    test_pred = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    # Save Model and artifacts (scaler/encoder)
    # We should save scaler and encoder too for the app
    artifacts = {
        'model': clf,
        'scaler': scaler,
        'label_encoder': le,
        'features': X.columns.tolist()
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Model and artifacts saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
