import streamlit as st
import pandas as pd

def sidebar_input_features():
    """
    Create sidebar inputs for student features.
    Returns a dataframe of the inputs.
    """
    st.sidebar.header('User Input Features')
    
    attendance = st.sidebar.slider(
        'Attendance Percentage',
        min_value=0, max_value=100, value=75, step=1
    )
    
    study_hours = st.sidebar.slider(
        'Study Hours per Day',
        min_value=0, max_value=24, value=5, step=1
    )
    
    assignments = st.sidebar.slider(
        'Assignments Completed Percentage',
        min_value=0, max_value=100, value=80, step=1
    )
    
    previous_score = st.sidebar.slider(
        'Previous Exam Score',
        min_value=0, max_value=100, value=70, step=1
    )
    
    data = {
        'attendance_percentage': attendance,
        'study_hours_per_day': study_hours,
        'assignments_completed_percentage': assignments,
        'previous_exam_score': previous_score
    }
    
    features = pd.DataFrame(data, index=[0])
    return features
