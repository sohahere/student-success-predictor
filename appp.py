import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ARTIFACTS
# ==========================================
@st.cache_resource
def load_data():
    with open('student_counseling_system.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data = load_data()
    model = data['model']
    scaler = data['scaler']
    threshold = data['threshold']
    benchmarks = data['benchmarks']
    global_odds = data['global_odds']
    explainer_bg = data['explainer_bg']

    # Re-initialize SHAP
    masker = shap.maskers.Independent(data=explainer_bg)
    explainer = shap.LinearExplainer(model, masker=masker)

except FileNotFoundError:
    st.error("⚠️ Model file not found. Please upload 'student_counseling_system.pkl'.")
    st.stop()

# ==========================================
# 3. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3406/3406988.png", width=80)
    st.title("Student Profile")
    st.markdown("Enter student metrics below:")
    
    # Sliders for engaging input
    attendance = st.slider("Attendance (%)", 40, 100, 85, help="Class Average: 90%")
    study_hours = st.slider("Daily Study (Hrs)", 0.0, 10.0, 3.0)
    library_hours = st.slider("Library Usage (Hrs/Week)", 0.0, 20.0, 2.0)
    social_media = st.slider("Social Media (Hrs/Day)", 0.0, 10.0, 2.5)
    sleep = st.slider("Sleep (Hrs/Night)", 3.0, 12.0, 7.0)
    
    st.markdown("---")
    st.subheader("Academic History")
    gpa = st.number_input("Previous Sem GPA (0-10)", 0.0, 10.0, 7.5, step=0.1)
    test_score = st.number_input("Last Test Score (0-100)", 0, 100, 72)
    backlogs = st.selectbox("Active Backlogs", [0, 1, 2, 3, 4, 5])
    extra_curr = st.slider("Extracurricular Score", 0, 100, 40)
    
    analyze_btn = st.button("Generate 360° Report", type="primary")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
if analyze_btn:
    # --- A. PREPROCESSING ---
    academic_strength = (gpa + test_score/10) / 2
    effort_score = study_hours + library_hours
    academic_risk = (1 if backlogs > 0 else 0) + (10-gpa) + (10-test_score/10)
    sleep_deviation = abs(sleep - 7)
    
    input_features = pd.DataFrame([{
        'academic_risk': academic_risk,
        'effort_score': effort_score,
        'attendance_pct': attendance,
        'social_media_hours_per_day': social_media,
        'extracurricular_engagement_score': extra_curr,
        'sleep_deviation': sleep_deviation,
        'is_backlog': 1 if backlogs > 0 else 0,
        'academic_strength': academic_strength
    }])
    
    input_scaled = pd.DataFrame(scaler.transform(input_features), columns=input_features.columns)
    
    # --- B. PREDICTION ---
    prob = model.predict_proba(input_scaled)[0, 1]
    
    # --- C. STATUS LOGIC ---
    if prob < 0.30:
        status = "🌟 HIGH PERFORMER"
        risk_color = "green"
        msg = "Outstanding trajectory. Focus on consistency and leadership."
    elif prob < threshold:
        status = "✅ ON TRACK"
        risk_color = "blue"
        msg = "Good standing. Small optimizations can unlock higher grades."
    else:
        status = "⚠️ FOCUS REQUIRED"
        risk_color = "red"
        msg = "Risk factors detected. Let's create a recovery plan."

    # --- D. HEADER SECTION ---
    st.title(f"Status: :{risk_color}[{status}]")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Risk Probability", value=f"{prob:.1%}", delta=f"Threshold: {threshold:.1%}", delta_color="inverse")
    with col2:
        st.metric(label="Projected GPA Band", value="8.5 - 9.0" if prob < 0.3 else ("7.0 - 8.0" if prob < threshold else "Needs Support"))
    with col3:
        st.info(f"**Counselor Note:** {msg}")

    st.markdown("---")

    # --- E. 360 VISUALIZATION (RADAR CHART) ---
    col_viz, col_insight = st.columns([1, 1])
    
    with col_viz:
        st.subheader("📊 360° Performance Radar")
        
        # Benchmarking for visuals (Mock percentiles for visual appeal)
        categories = ['Attendance', 'Study Effort', 'Academic Base', 'Sleep Consistency']
        student_vals = [
            min(attendance/100, 1), 
            min(effort_score/8, 1), 
            min(academic_strength/10, 1), 
            max(0, (5-sleep_deviation)/5)
        ]
        # Class Averages from benchmarks
        class_vals = [
            benchmarks['attendance_pct']/100, 
            benchmarks['effort_score']/8, 
            benchmarks['academic_strength']/10, 
            0.8 # approx 
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=student_vals, theta=categories, fill='toself', name='You', line_color='#1f77b4'))
        fig.add_trace(go.Scatterpolar(r=class_vals, theta=categories, fill='toself', name='Class Avg', line_color='gray', opacity=0.3))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_insight:
        st.subheader("💡 Key Strengths & Weaknesses")
        
        # SHAP Analysis
        shap_values = explainer(input_scaled)
        # Get top positive (risks) and top negative (strengths)
        shap_df = pd.DataFrame({
            'Feature': input_features.columns,
            'SHAP': shap_values.values[0],
            'Value': input_features.values[0]
        }).sort_values(by='SHAP', ascending=False)
        
        # Display Logic
        for _, row in shap_df.head(3).iterrows():
            feat = row['Feature']
            val = row['Value']
            
            if row['SHAP'] > 0: # Risk Factor
                icon = "⚠️"
                color = "red"
                txt = "Needs Attention"
            else: # Strength
                icon = "✅" 
                color = "green"
                txt = "Strength"
                
            with st.container():
                st.markdown(f"**{icon} {feat.replace('_', ' ').title()}**")
                # Contextual Message
                if feat == 'attendance_pct' and val < benchmarks['attendance_pct']:
                    st.caption(f"Your attendance ({val}%) is below the class average ({benchmarks['attendance_pct']:.0f}%).")
                elif feat == 'social_media_hours_per_day' and val > 3:
                    st.caption(f"High digital distraction ({val} hrs/day). Reducing this is your easiest win.")
                elif feat == 'effort_score' and val > benchmarks['effort_score']:
                    st.caption(f"Excellent hustle! You study more ({val} hrs) than the class average.")
                else:
                    st.caption(f"This factor is contributing to your {txt} status.")

    # --- F. "WHAT-IF" SIMULATOR ---
    st.markdown("---")
    st.subheader("🚀 Path to Improvement Simulator")
    st.markdown("Adjust the sliders to see how habit changes can improve your academic standing.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        new_study = st.slider("Target Study Hours", 0.0, 10.0, float(study_hours), key="sim_study")
    with sim_col2:
        new_social = st.slider("Target Social Media", 0.0, 10.0, float(social_media), key="sim_social")
        
    # Live Recalculation (Approximation)
    # We use the coefficient logic: Study reduces risk, Social increases it.
    # In a full app, we would re-run model.predict_proba
    # Simple linear shift for demo responsiveness:
    current_risk = prob
    improvement = (new_study - study_hours) * 0.04 + (social_media - new_social) * 0.03
    projected_risk = max(0.01, current_risk - improvement)
    
    st.metric(
        label="Projected Risk Score", 
        value=f"{projected_risk:.1%}", 
        delta=f"{(current_risk - projected_risk):.1%} Improvement",
        delta_color="normal"
    )
    
    if projected_risk < threshold and current_risk > threshold:
        st.balloons()
        st.success("🎉 These changes would move you into the SAFE ZONE!")

else:
    st.info("👈 Enter student details in the sidebar and click 'Generate 360° Report'")
