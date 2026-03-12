import pandas as pd

def feature_engineering(df):
    """
    Apply feature engineering to the dataset.
    """
    df = df.copy()
    
    # Example: Interaction term
    # Study Efficiency = Study Hours * Attendance (proxy for engagement)
    # We normalize attendance to 0-1 for this Interaction
    df['engagement_score'] = df['study_hours_per_day'] * (df['attendance_percentage'] / 100.0)
    
    # Average performance metric (Assignments + Previous Exam)
    df['avg_performance_metric'] = (df['assignments_completed_percentage'] + df['previous_exam_score']) / 2.0
    
    print("Feature engineering applied: 'engagement_score', 'avg_performance_metric' added.")
    return df
