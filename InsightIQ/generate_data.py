import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset size
num_samples = 1000

# Generate features
attendance = np.random.randint(60, 100, num_samples)
study_hours = np.random.randint(1, 10, num_samples)
assignments = np.random.randint(50, 100, num_samples)
previous_score = np.random.randint(40, 100, num_samples)

# Calculate weighted score with some noise
# Formula: 20% attendance, 20% study (scaled), 20% assignments, 40% previous
noise = np.random.normal(0, 5, num_samples)
final_score = (0.2 * attendance) + (0.2 * study_hours * 10) + (0.2 * assignments) + (0.4 * previous_score) + noise

# Define grade based on score
def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 75: return 'B'
    elif score >= 60: return 'C'
    else: return 'D'

grades = [get_grade(s) for s in final_score]

# Create DataFrame
df = pd.DataFrame({
    'attendance_percentage': attendance,
    'study_hours_per_day': study_hours,
    'assignments_completed_percentage': assignments,
    'previous_exam_score': previous_score,
    'final_grade': grades
})

# Save to CSV
output_path = r'd:\INIQ\InsightIQ\data\raw\student_performance_raw.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset generated at {output_path}")
print(df.head())
print(df['final_grade'].value_counts())
