import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import time

# ==========================================
# 1. VISUAL CONFIGURATION (The "Pretty" Part)
# ==========================================
st.set_page_config(
    page_title="Student Success Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Fonts, Cards, and Badges
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #ccc;
    }
    
    .card-risk-high { border-left-color: #ff4b4b; }
    .card-risk-med { border-left-color: #ffa726; }
    .card-risk-low { border-left-color: #4CAF50; }
    
    /* Typography */
    h1 { color: #1e293b; font-weight: 700; }
    h3 { color: #334155; font-weight: 600; font-size: 1.1rem; }
    p { color: #475569; font-size: 0.95rem; line-height: 1.5; }
    
    /* Custom Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-green { background-color: #dcfce7; color: #166534; }
    .badge-red { background-color: #fee2e2; color: #991b1b; }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD THE BRAIN
# ==========================================
@st.cache_resource
def load_data():
    # Load your pickle file
    try:
        with open('student_counseling_system.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("⚠️ System Error: 'student_counseling_system.pkl' not found. Please run the Training Notebook first.")
        st.stop()

data = load_data()
model = data['model']
scaler = data['scaler']
threshold = data['threshold']
benchmarks = data['benchmarks']
global_odds = data['global_odds'] # Used for global context context

# ==========================================
# 3. HELPER: HUMAN LANGUAGE GENERATOR
# ==========================================
def interpret_attendance(val, benchmark):
    gap = benchmark - val
    if val >= benchmark:
        return f"🌟 **Elite Consistency:** You attend {val}% of classes. This is your superpower. Keep showing up, and the grades will follow."
    elif val >= 85:
        return f"✅ **Solid Zone:** 85% is safe, but the class average is higher ({benchmark:.0f}%). Attending just 1 more class per week puts you in the top tier."
    else:
        missed_classes = int((benchmark - val) / 5) # Approx 5% = 1 class/week
        return f"⚠️ **The Silent Grade Killer:** You are at {val}%. In real terms, you are missing ~{missed_classes} more classes per week than your peers. Closing this specific gap is the easiest way to improve."

def interpret_social_media(val):
    if val <= 2:
        return f"🧠 **Deep Focus Advantage:** Keeping usage under 2 hours ({val} hrs) gives your brain a huge recovery advantage over classmates."
    elif val <= 3.5:
        return f"📱 **Average Distraction:** {val} hours is normal, but 'normal' students get average grades. Cutting 30 mins here creates 30 mins for sleep or study."
    else:
        return f"🛑 **Dopamine Drain:** {val} hours/day is ~{val*7} hours/week. That's a full-time job! Imagine the GPA boost if you re-invested just half of that time."

def interpret_backlog(val):
    if val == 0:
        return "✨ **Clear Path:** No backlogs means zero 'academic debt'. You can focus 100% on new topics."
    else:
        return "🚧 **Roadblock Detected:** A backlog is like running a race with a backpack. Prioritize clearing this above *everything* else."

# ==========================================
# 4. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4762/4762311.png", width=60)
    st.title("Student Profile")
    st.markdown("Update metrics to refresh the report.")
    
    st.caption("LIFESTYLE METRICS")
    attendance = st.slider("Attendance", 40, 100, 85, format="%d%%")
    social_media = st.slider("Social Media (Hrs/Day)", 0.0, 8.0, 3.0, step=0.5)
    sleep = st.slider("Sleep (Hrs/Night)", 3.0, 10.0, 6.5, step=0.5)
    
    st.caption("ACADEMIC METRICS")
    study_hours = st.number_input("Self Study (Hrs/Day)", 0.0, 10.0, 2.0)
    library_hours = st.number_input("Library (Hrs/Week)", 0.0, 20.0, 2.0)
    gpa = st.number_input("Prev. Sem GPA", 0.0, 10.0, 7.2)
    test_score = st.number_input("Last Test Score", 0, 100, 68)
    backlogs = st.radio("Active Backlogs?", [0, 1, 2, 3], horizontal=True)
    extra_curr = st.slider("Extracurricular Score", 0, 100, 40)

# ==========================================
# 5. REAL-TIME PREDICTION ENGINE
# ==========================================

# 1. Feature Engineering (Live)
effort_score = study_hours + library_hours
academic_strength = (gpa + test_score/10) / 2
academic_risk = (1 if backlogs > 0 else 0) + (10-gpa) + (10-test_score/10)
sleep_deviation = abs(sleep - 7)

# 2. Prepare Dataframe
input_df = pd.DataFrame([{
    'academic_risk': academic_risk,
    'effort_score': effort_score,
    'attendance_pct': attendance,
    'social_media_hours_per_day': social_media,
    'extracurricular_engagement_score': extra_curr,
    'sleep_deviation': sleep_deviation,
    'is_backlog': 1 if backlogs > 0 else 0,
    'academic_strength': academic_strength
}])

# 3. Scale & Predict
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
prob = model.predict_proba(input_scaled)[0, 1]

# 4. Determine Status
if prob < 0.30:
    status_title = "High Potential"
    status_color = "card-risk-low"
    status_msg = "You are in the **Safe Zone**. Your habits align with high achievers."
    bar_color = "#4CAF50"
elif prob < threshold:
    status_title = "On Track"
    status_color = "card-risk-med"
    status_msg = "You are doing well, but slightly **inconsistent**. A few tweaks will secure your grades."
    bar_color = "#ffa726"
else:
    status_title = "Growth Opportunity" # Never say "At Risk" in title
    status_color = "card-risk-high"
    status_msg = "Current trends suggest **academic pressure ahead**. Let's adjust the strategy now."
    bar_color = "#ff4b4b"

# ==========================================
# 6. MAIN DASHBOARD LAYOUT
# ==========================================

st.title(f"👋 Your Personalized Success Roadmap")
st.markdown(f"**Analysis based on Class Benchmarks & Historical Trends**")

# --- SECTION A: STATUS CARD ---
st.markdown(f"""
<div class="metric-card {status_color}">
    <h2 style="margin:0">{status_title}</h2>
    <p style="font-size:18px; margin-top:5px;">{status_msg}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 Performance Radar")
    # Radar Chart
    categories = ['Attendance', 'Study Effort', 'Grades', 'Sleep Consistency']
    student_vals = [
        min(attendance/100, 1), 
        min(effort_score/10, 1), 
        min(academic_strength/10, 1), 
        max(0, (5-sleep_deviation)/5)
    ]
    # Benchmarks (Mocked for visual contrast based on saved medians)
    class_vals = [0.92, 0.5, 0.75, 0.8] 
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=student_vals, theta=categories, fill='toself', name='You', line_color=bar_color))
    fig.add_trace(go.Scatterpolar(r=class_vals, theta=categories, fill='toself', name='Class Median', line_color='gray', opacity=0.3, line_dash='dot'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=True, margin=dict(t=20, b=20, l=40, r=40), height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Deep Dive & Coaching")
    
    # 1. ATTENDANCE INSIGHT
    with st.expander("📅 Attendance Analysis", expanded=True):
        st.markdown(interpret_attendance(attendance, benchmarks['attendance_pct']))
        
    # 2. SOCIAL MEDIA INSIGHT
    with st.expander("📱 Digital Focus", expanded=True):
        st.markdown(interpret_social_media(social_media))
        
    # 3. BACKLOG INSIGHT
    if backlogs > 0:
        with st.expander("🚧 Academic Roadblocks", expanded=True):
            st.markdown(interpret_backlog(backlogs))

# --- SECTION B: THE "REAL" SIMULATOR ---
st.markdown("---")
st.subheader("🚀 'What-If' Simulator")
st.markdown("Drag the sliders below to simulate a new reality. **The chart updates in real-time.**")

sim_col1, sim_col2, sim_col3 = st.columns(3)

with sim_col1:
    new_study = st.slider("Target: Daily Study (Hrs)", 0.0, 10.0, float(study_hours))
with sim_col2:
    new_social = st.slider("Target: Social Media (Hrs)", 0.0, 8.0, float(social_media))
with sim_col3:
    new_attend = st.slider("Target: Attendance (%)", 40, 100, int(attendance))

# --- LIVE SIMULATION LOGIC ---
# We recreate the input vector with NEW slider values
sim_effort = new_study + library_hours
sim_input = input_df.copy()
sim_input['effort_score'] = sim_effort
sim_input['social_media_hours_per_day'] = new_social
sim_input['attendance_pct'] = new_attend

# Predict New Probability
sim_scaled = pd.DataFrame(scaler.transform(sim_input), columns=input_df.columns)
new_prob = model.predict_proba(sim_scaled)[0, 1]

# Calculate Improvement
improvement = prob - new_prob

# VISUALIZE THE CHANGE
st.metric(
    label="Projected Risk Probability",
    value=f"{new_prob:.1%}",
    delta=f"{(prob - new_prob):.1%} Improvement",
    delta_color="inverse" # Green is good for risk reduction
)

# Encouragement Logic
if new_prob < threshold and prob > threshold:
    st.success("🎉 **Success!** With these habits, you move from the **Growth Zone** to **On Track**!")
    st.balloons()
elif improvement > 0.05:
    st.info(f"💪 Great job! These changes reduce your academic risk by **{improvement:.1%}**. Keep optimizing!")
elif improvement <= 0:
    st.warning("Try increasing study hours or attendance to see a positive impact.")
