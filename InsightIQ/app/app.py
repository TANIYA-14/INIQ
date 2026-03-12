import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from feature_engineering import feature_engineering

# Load Model
@st.cache_resource
def load_model_artifacts():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'student_performance_model.pkl')
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

def get_grade_style(grade):
    """Returns style properties for a given grade."""
    styles = {
        'A': {'color': '#4CAF50', 'icon': '🎓', 'label': 'Excellent'},  # Green
        'B': {'color': '#2196F3', 'icon': '📊', 'label': 'Good'},       # Blue
        'C': {'color': '#FF9800', 'icon': '📉', 'label': 'Average'},    # Orange
        'D': {'color': '#F44336', 'icon': '⚠️', 'label': 'Poor'}        # Red
    }
    return styles.get(grade, {'color': '#808080', 'icon': '❓', 'label': 'Unknown'})

def generate_study_plan(grade, attendance, study_hours, assignments):
    """Generates a personalized study plan based on grade and metrics."""
    plan = []

    # Base Routine per Grade
    if grade == 'A':
        plan.extend([
            "🌟 **Maintain Consistency:** Keep up your current effective study habits.",
            "⏱️ **Daily Revision:** Dedicate 30 minutes to review key concepts daily.",
            "📝 **Mock Tests:** Attempt one full-length mock test weekly to ensure retention."
        ])
    elif grade == 'B':
        plan.extend([
            "📚 **Focused Study:** Increase daily study time to 2–3 hours.",
            "🎯 **Weak Areas:** Identify and strictly focus on 1 or 2 weak subjects.",
            "🔄 **Regular Revision:** Review weekly notes every weekend."
        ])
    elif grade == 'C':
        plan.extend([
            "📖 **Core Fundamentals:** Study 3–4 hours daily focusing on basic concepts.",
            "✍️ **Assignments:** prioritize completing all pending assignments.",
            "❓ **Ask Questions:** Don't hesitate to seek help from teachers or peers on tough topics."
        ])
    else: # Grade D
        plan.extend([
            "🔥 **Intensive Study:** Dedicate 4–5 hours daily; this is critical.",
            "📅 **Strict Schedule:** Create and stick to a rigid hourly study planner.",
            "🆘 **Academic Support:** Immediately seek extra tutoring or support groups."
        ])

    # Dynamic Personalization using Input Metrics
    if attendance < 50:
        plan.append("⚠️ **Attendance Alert:** Your attendance is critically low. Aim to attend every single class moving forward.")
    
    if study_hours < 2:
        plan.append("⏰ **Time Management:** You are studying less than 2 hours. Try using the Pomodoro technique (25m study, 5m break) to build a habit.")
        
    if assignments < 60:
        plan.append("📝 **Assignment Priority:** Submission rate is low. Submit all backlog assignments immediately to boost internal assessment scores.")

    return plan

def add_custom_css():
    """Injects CSS for 3D animated background and glassmorphism."""
    st.markdown("""
    <style>
    /* 3D Animated Gradient Background */
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Glassmorphism for Main Container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.05); /* Semi-transparent white */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 3rem;
        margin-top: 2rem;
    }

    /* Typography Visibility */
    h1, h2, h3, h4, int, p, label, .stMarkdown, .stDataFrame {
        color: #ffffff !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Input Widgets */
    .stSlider > div > div > div > div {
        color: #fff;
    }

    /* Result Card Styling in Dark Mode */
    .result-card {
        background: rgba(0, 0, 0, 0.3); 
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="InsightIQ - Student Performance", layout="wide")
    
    # Inject 3D Background & Glassmorphism
    add_custom_css()
    
    st.title("InsightIQ – AI-Based Student Performance Predictor")
    st.write("""
    This app predicts the **Grade** of a student based on their academic behaviors and previous records.
    """)
    
    # Initialize Session State
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Import UI
    try:
        from ui_components import sidebar_input_features
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ui_components import sidebar_input_features

    # Sidebar inputs
    input_df = sidebar_input_features()
    
    # Validation Warnings
    if input_df['attendance_percentage'].iloc[0] < 50:
        st.warning("⚠️ Attendance below 50% may seriously affect performance.")
    if input_df['study_hours_per_day'].iloc[0] < 2:
        st.warning("📉 Studying less than 2 hours per day may reduce grades.")
    if input_df['assignments_completed_percentage'].iloc[0] < 60:
        st.warning("📝 Low assignment completion can impact final results.")

    st.subheader('User Input parameters')
    st.write(input_df)
    
    # Load Model
    artifacts = load_model_artifacts()
    
    if artifacts is None:
        st.error("Model not found. Please train the model first by running `src/model_training.py`.")
        return

    clf = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['label_encoder']
    feat_cols_order = artifacts['features']
    
    if st.button('Predict'):
        try:
             # Feature Engineering
             processed_df = feature_engineering(input_df)
             
             # Align Columns
             missing_cols = set(feat_cols_order) - set(processed_df.columns)
             if missing_cols:
                 st.error(f"Missing columns: {missing_cols}")
                 return
             
             X_input = processed_df[feat_cols_order]
             
             # Scaling
             num_cols = X_input.select_dtypes(include=['int64', 'float64']).columns
             X_input_scaled = X_input.copy()
             X_input_scaled[num_cols] = scaler.transform(X_input[num_cols])
             
             # Prediction
             prediction = clf.predict(X_input_scaled)
             prediction_proba = clf.predict_proba(X_input_scaled)
             
             # Decode
             predicted_grade = le.inverse_transform(prediction)[0]
             
             # Style Result
             style = get_grade_style(predicted_grade)
             
             # Result Card
             st.markdown(f"""
             <div class="result-card" style="padding: 20px; border-radius: 10px; border-left: 5px solid {style['color']};">
                 <h2 style="color: {style['color']} !important; margin:0;">
                     {style['icon']} Predicted Grade: {predicted_grade}
                 </h2>
                 <p style="font-size: 18px; margin: 5px 0 0 0; color: #fff !important;">
                     <strong>Result:</strong> {style['label']}
                 </p>
             </div>
             """, unsafe_allow_html=True)
             
             # Update History
             new_record = input_df.copy()
             new_record['Predicted Grade'] = predicted_grade
             st.session_state['history'].insert(0, new_record) # Newest first
             
             # Keep only last 5
             if len(st.session_state['history']) > 5:
                 st.session_state['history'] = st.session_state['history'][:5]
             
             # Personalized Study Routine
             st.markdown("### 📅 Personalized Study Routine")
             
             # Get inputs for logic
             att = input_df['attendance_percentage'].iloc[0]
             study = input_df['study_hours_per_day'].iloc[0]
             assign = input_df['assignments_completed_percentage'].iloc[0]
             
             study_plan = generate_study_plan(predicted_grade, att, study, assign)
             
             for item in study_plan:
                 st.markdown(f"- {item}")
             
             st.subheader('Prediction Probability')
             proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
             st.bar_chart(proba_df.T)
             
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
    # History Table
    if st.session_state['history']:
        st.subheader("Prediction History (Last 5)")
        # Concat list of DFs
        history_df = pd.concat(st.session_state['history'], ignore_index=True)
        st.dataframe(history_df)

    # Visualizations (Feature Importance etc.)
    st.subheader("Model Insights")
    col1, col2 = st.columns(2)
    
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    cm_path = os.path.join(report_dir, 'confusion_matrix.png')
    fi_path = os.path.join(report_dir, 'feature_importance.png')
    
    with col1:
        if os.path.exists(fi_path):
            st.image(fi_path, caption='Feature Importance')
    with col2:
        if os.path.exists(cm_path):
            st.image(cm_path, caption='Confusion Matrix')

if __name__ == "__main__":
    main()
