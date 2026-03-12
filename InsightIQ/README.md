# InsightIQ – AI-Based Student Performance Predictor

## Problem Statement
Predicting student academic performance is crucial for early intervention. This project aims to predict student grades based on study habits and previous performance metrics.

## Solution Overview
InsightIQ is an end-to-end machine learning system that uses a Random Forest Classifier to predict grades (A, B, C, D) based on:
- Attendance Percentage
- Study Hours per Day
- Assignments Completed Percentage
- Previous Exam Scores

The project includes a Streamlit web interface for easy interaction.

## Folder Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: EDA and experiments.
- `src/`: Source code for preprocessing, training, and evaluation.
- `models/`: Serialized models.
- `app/`: Streamlit application.
- `reports/`: Evaluation metrics and plots.

## Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit
- Matplotlib/Seaborn

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data generation (optional): `python generate_data.py`
3. Train model: `python src/model_training.py`
4. Run app: `streamlit run app/app.py`
