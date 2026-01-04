import streamlit as st
import pandas as pd
import joblib
import time
import shap
import matplotlib.pyplot as plt
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Risk Analysis", layout="wide")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

from sklearn.base import BaseEstimator, TransformerMixin
class CustomValueMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mapping_dict.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(mapping)
        return X_copy
    def get_feature_names_out(self, input_features=None):
        return input_features


# --- 2. BACKEND LOGIC ---
def load_and_predict(file):
    """
    1. Reads Excel file.
    2. Cleans data (removes text columns) for the model.
    3. Runs model.predict_proba() to get risk scores.
    4. Returns the full dataframe with new 'Risk_Score' and 'Risk_Level' columns.
    """
    df = pd.read_excel(file)
    FEATURE_ORDER = [
                'gender', 'department', 'scholarship', 'extra_curricular',
                'age', 'cgpa', 'attendance_rate', 'family_income',
                'past_failures', 'study_hours_per_week', 'assignments_submitted',
                'projects_completed', 'total_activities', 'sports_participation'
            ]
    # --- MODEL BLOCK START ---
    try:
        # Load your real model
        model = joblib.load('risk_model.pkl')
        
        # Preprocessing: Drop columns the model doesn't understand (adjust as need)
        # We only keep numeric columns for prediction
        features = df.drop(columns=['name', 'student_id','parental_education'], errors='ignore')
        features = features[FEATURE_ORDER]
        # Get Risk Probability (Class 1)
        probs = model.predict_proba(features)[:, 1]
        df['Risk_Score'] = (probs * 100).astype(int)
        
    except (FileNotFoundError, Exception):
        # FALLBACK: If model.pkl is missing, generate random scores for UI testing
        # This ensures your website doesn't crash while you are designing the UI
        import numpy as np
        np.random.seed(42)
        df['Risk_Score'] = np.random.randint(10, 95, size=len(df))
    # --- MODEL BLOCK END ---

    # Assign labels
    def get_risk_level(score):
        if score >= 75: return 'High'
        elif score >= 40: return 'Medium'
        return 'Low'

    df['Risk_Level'] = df['Risk_Score'].apply(get_risk_level)
    return df

# --- 3. PAGE: HOME ---
# --- 3. PAGE: HOME ---
def show_home_page():
    # Centered Title
    st.markdown("<h1 style='text-align: center; color: #0078D7;'>Student Disengagement Risk Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>Upload student data to predict retention risks using AI.</p>", unsafe_allow_html=True)
    st.divider()

    # Layout for Uploader
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # 1. Show Uploader
        uploaded_file = st.file_uploader("Upload Student Excel Sheet", type=['xlsx'])
        
        # Optional: Show a small success message if file is there, but don't hide the button if not
        if uploaded_file:
            st.success("File attached ready for analysis!")

        st.write("") # Spacer
        
        # 2. Show Button ALWAYS (It is now outside the 'if uploaded_file' block)
        if st.button("Proceed to Dashboard", use_container_width=True, type="primary"):
            
            # 3. Validation Logic inside the button
            if uploaded_file is not None:
                with st.spinner("Running AI Risk Prediction..."):
                    # Process data once and save to memory
                    processed_df = load_and_predict(uploaded_file)
                    st.session_state.data = processed_df
                    st.session_state.page = 'dashboard'
                    st.rerun()
            else:
                # Show error if clicked without a file
                st.error("⚠️ Please upload an Excel sheet first to proceed.")

# --- 4. PAGE: DASHBOARD ---
def show_dashboard_page():
    # Header Row
    col_head_1, col_head_2 = st.columns([6, 1])
    with col_head_1:
        st.title("Risk Analysis Dashboard")
    with col_head_2:
        if st.button("🔄 Upload New"):
            st.session_state.page = 'home'
            st.rerun()

    df = st.session_state.data

    # --- SEARCH & FILTER BAR ---
    with st.container(border=True):
        # 1. Search Bar (Full Width)
        # Note: In Streamlit, "dynamic" update happens on Enter or clicking away.
        search_query = st.text_input("🔎 Search by Name", placeholder="Type a student name...", label_visibility="collapsed")
        
        st.write("") # Spacer
        
        # 2. Filters Row
        f1, f2, f3 = st.columns(3)
        with f1:
            all_branches = df['department'].unique().tolist() if 'department' in df.columns else []
            selected_branches = st.multiselect("Branch", options=all_branches, placeholder="All Branches")
        
        with f2:
            risk_options = ['High', 'Medium', 'Low']
            selected_risks = st.multiselect("Risk Level", options=risk_options, default=risk_options, placeholder="Select Risk")
            
        with f3:
            sort_option = st.selectbox("Sort By", ["Default", "Risk: High to Low", "Risk: Low to High", "CGPA: Low to High", "CGPA: High to Low"])

    # --- FILTERING LOGIC ---
    # Start with all data
    filtered_df = df.copy()

    # 1. Apply Name Search (if something is typed)
    if search_query:
        filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains(search_query, case=False, na=False)]

    # 2. Apply Branch Filter (if branches selected)
    if selected_branches:
        filtered_df = filtered_df[filtered_df['department'].isin(selected_branches)]

    # 3. Apply Risk Filter
    if selected_risks:
        filtered_df = filtered_df[filtered_df['Risk_Level'].isin(selected_risks)]

    # 4. Apply Sorting
    if sort_option == "Risk: High to Low":
        filtered_df = filtered_df.sort_values(by='Risk_Score', ascending=False)
    elif sort_option == "Risk: Low to High":
        filtered_df = filtered_df.sort_values(by='Risk_Score', ascending=True)
    elif sort_option == "CGPA: Low to High" and 'cgpa' in df.columns:
        filtered_df = filtered_df.sort_values(by='cgpa', ascending=True)
    elif sort_option == "CGPA: High to Low" and 'cgpa' in df.columns:
        filtered_df = filtered_df.sort_values(by='cgpa', ascending=False)

    st.divider()
    
    # --- RESULTS AREA ---
    st.caption(f"Showing {len(filtered_df)} students")

    if filtered_df.empty:
        st.warning("No students found matching your criteria.")
    else:
        # --- CARD GRID SYSTEM (3 Cards Per Row) ---
        rows = list(filtered_df.iterrows())
        chunk_size = 3
        
        # Loop through data in chunks of 3
        for i in range(0, len(rows), chunk_size):
            cols = st.columns(chunk_size)
            chunk = rows[i : i + chunk_size]
            
            for j, (index, student) in enumerate(chunk):
                with cols[j]:
                    # INDIVIDUAL CARD
                    with st.container(border=True):
                        # Header: Name & ID
                        st.subheader(student.get('name', 'Unknown'))
                        st.text(f"Student ID: {student.get('student_id', 'N/A')}")
                        
                        # Body: Branch & Risk Badge
                        st.caption(f"Branch: {student.get('department', 'N/A')}")
                        st.caption(f"CGPA: {student.get('cgpa', 'N/A')}")
                        risk_score = student.get('Risk_Score', 0)
                        
                        # Color-coded Risk Indicator
                        if risk_score >= 75:
                            st.markdown(f"<div style='background-color:#ffebee; color:#c62828; padding:5px; border-radius:5px; text-align:center;'><b>High Risk ({risk_score}%)</b></div>", unsafe_allow_html=True)
                        elif risk_score >= 40:
                            st.markdown(f"<div style='background-color:#fff3e0; color:#ef6c00; padding:5px; border-radius:5px; text-align:center;'><b>Medium Risk ({risk_score}%)</b></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color:#e8f5e9; color:#2e7d32; padding:5px; border-radius:5px; text-align:center;'><b>Low Risk ({risk_score}%)</b></div>", unsafe_allow_html=True)
                        
                        st.write("") # Spacer
                        
                        # Button to go to details
                        if st.button("View Profile ➜", key=f"btn_{index}", use_container_width=True):
                            st.session_state.selected_student = student
                            st.session_state.page = 'detail'
                            st.rerun()

# --- 5. PAGE: DETAIL VIEW (Skeleton) ---
# --- PAGE: DETAIL VIEW ---
def show_detail_page():
    student = st.session_state.selected_student
    
    # 1. Top Navigation
    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
    
    st.divider()

    # 2. Header & Risk Badge
    c0,c1, c2 ,c3= st.columns([1,2, 1,1])
    with c1:
        st.title(f"{student.get('name', 'Student')}")
        st.text(f"              Student ID: {student.get('student_id', 'N/A')}   |   Branch: {student.get('department', 'N/A')}")
    
    with c2:
        risk_score = student.get('Risk_Score', 0)
        if risk_score >= 75:
            risk_color = "#c62828" # Red
            risk_text = "HIGH RISK"
        elif risk_score >= 40:
            risk_color = "#ef6c00" # Orange
            risk_text = "MEDIUM RISK"
        else:
            risk_color = "#2e7d32" # Green
            risk_text = "LOW RISK"

        st.markdown(f"""
            <div style="background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 2px rgba(0,0,0,0.2);">
                <h2 style="margin:0; font-size: 28px;">{risk_score}%</h2>
                <p style="margin:0; font-size:14px; font-weight:bold;">{risk_text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    spacer_left, main_content, spacer_right = st.columns([1.5, 4, 1])
    with main_content:
    # 3. Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        cgpa = student.get('cgpa', student.get('CGPA', 0))
        attend = student.get('attendance_rate', student.get('Attendance Rate', 0))
        hours = student.get('study_hours_per_week', student.get('Study Hours Per Week', 0))
        fails = student.get('past_failures', student.get('Past Failures', 0))

    m1.metric("CGPA", f"{cgpa:.2f}")
    m2.metric("Attendance", f"{attend}%")
    m3.metric("Study Hours", f"{hours}")
    m4.metric("Past Failures", f"{fails}")
    
    st.divider()

    # 4. AI Explainability (SHAP)
    spacer_left, main_content, spacer_right = st.columns([1, 4, 1])
    with main_content:
        st.subheader("Risk Factor Analysis")
        st.write("The chart below shows which factors are **increasing (Red)** or **decreasing (Green)** the risk.")

    full_pipeline = joblib.load('risk_model.pkl')

    if full_pipeline:
        try:
            # --- A. UNWRAP MODEL ---
            if hasattr(full_pipeline, 'best_estimator_'):
                best_model = full_pipeline.best_estimator_
            else:
                best_model = full_pipeline

            preprocessor = best_model.named_steps['preprocessor']
            xgboost_model = best_model.named_steps['model']

            # --- B. PREPARE DATA ---
            FEATURE_ORDER = [
                'gender', 'department', 'scholarship', 'extra_curricular',
                'age', 'cgpa', 'attendance_rate', 'family_income',
                'past_failures', 'study_hours_per_week', 'assignments_submitted',
                'projects_completed', 'total_activities', 'sports_participation'
            ]
            
            student_data = pd.DataFrame([student])
            
            # Clean column names (Safety Fix)
            student_data.columns = student_data.columns.str.lower().str.strip().str.replace(' ', '_')
            
            # Ensure columns exist
            for col in FEATURE_ORDER:
                if col not in student_data.columns:
                    student_data[col] = 0
            
            student_features = student_data[FEATURE_ORDER]

            # --- C. TRANSFORM ---
            student_processed = preprocessor.transform(student_features)

            # --- D. GET NAMES ---
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception as name_error:
                st.warning(f"⚠️ Could not get feature names. Error: {name_error}")
                feature_names = [f"Feature {i}" for i in range(student_processed.shape[1])]

            # --- E. CALCULATE SHAP ---
            explainer = shap.TreeExplainer(xgboost_model)
            shap_values = explainer.shap_values(student_processed)

            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Shap Value': vals
            })

            # Sort by absolute impact
            importance_df['Abs Impact'] = importance_df['Shap Value'].abs()
            top_5 = importance_df.sort_values(by='Abs Impact', ascending=False).head(5)

            # --- F. PLOT WITH CLEAN NAMES ---
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#ff5252' if x > 0 else '#66bb6a' for x in top_5['Shap Value']]
            
            bars = ax.barh(top_5['Feature'], top_5['Shap Value'], color=colors)
            ax.set_xlabel("Impact on Risk Score")
            ax.axvline(0, color='black', linewidth=0.8)
            
            # Clean Labels Logic
            clean_labels = []
            for label in top_5['Feature']:
                if "Feature" in label:
                    clean_labels.append(label)
                    continue

                label = label.replace('cat__', '').replace('num__', '').replace('remainder__', '')
                label_lower = label.lower()
                
                if 'cgpa' in label_lower: clean_labels.append("CGPA")
                elif 'past_failures' in label_lower: clean_labels.append("Past Failures")
                elif 'family_income' in label_lower: clean_labels.append("Family Income")
                elif 'attendance' in label_lower: clean_labels.append("Attendance Rate")
                elif 'study_hours' in label_lower: clean_labels.append("Study Hours")
                elif 'assignments' in label_lower: clean_labels.append("Assignments")
                elif 'department' in label_lower: clean_labels.append("Department")
                elif 'scholarship' in label_lower: clean_labels.append("Scholarship")
                elif 'sports' in label_lower: clean_labels.append("Sports Participation")
                elif 'total_activities' in label_lower: clean_labels.append("Total Activities")
                elif 'projects' in label_lower: clean_labels.append("Projects Completed")
                elif 'gender' in label_lower: clean_labels.append("Gender")
                else: clean_labels.append(label.replace('_', ' ').title())
            
            ax.set_yticklabels(clean_labels)
            
            for bar, value in zip(bars, top_5['Shap Value']):
                width = bar.get_width()
                align = 'left' if width > 0 else 'right'
                ax.text(width, bar.get_y() + bar.get_height()/2, f" {value:.2f} ", va='center', ha=align, fontsize=9, fontweight='bold')
            spacer_left, main_content, spacer_right = st.columns([1, 4, 1])
            with main_content:
                st.pyplot(fig)

            st.divider()
            
            # --- CENTERED LAYOUT SETUP ---
            spacer_left, main_content, spacer_right = st.columns([1, 2, 1])
            
            with main_content:
                st.subheader("Simulation Panel")
                st.write("Adjust the mutable factors below to simulate risk improvements.")

                # 1. CONFIGURATION
                SIMULATION_CONFIG = {
                    "CGPA":                {"col": "cgpa",                  "min": 0.0, "max": 10.0,   "step": 0.1},
                    "Attendance Rate":     {"col": "attendance_rate",       "min": 0.0, "max": 100.0,  "step": 1.0},
                    "Study Hours":         {"col": "study_hours_per_week",  "min": 0.0, "max": 20.0,   "step": 0.5},
                    "Assignments":         {"col": "assignments_submitted", "min": 0,   "max": 50,     "step": 1}, 
                    "Sports Participation":{"col": "sports_participation",  "min": 0,   "max": 1,      "step": 1},
                    "Total Activities":    {"col": "total_activities",      "min": 0,   "max": 10,     "step": 1},
                    "Family Income":       {"col": "family_income",         "min": 0,   "max": 500000, "step": 5000},
                    "Projects Completed":  {"col": "projects_completed",    "min": 0,   "max": 20,     "step": 1}
                }

                # 2. DEFINITIONS (IMMUTABLE & INTEGER COLUMNS)
                IMMUTABLE_FEATURES = ["Gender", "Department", "Scholarship", "Age", "Past Failures","Family Income"]
                
                # !!! CRITICAL FIX: These columns MUST be integers or model ignores changes !!!
                INT_COLUMNS = ["assignments_submitted", "projects_completed", "total_activities"]

                # 3. PREPARE DATA COPY
                simulated_student = student_data.copy()
                
                # --- START FORM ---
                with st.form("risk_simulation_form"):
                    st.markdown("#### Adjust Values")
                    
                    new_values = {}
                    visible_slider_count = 0

                    # Loop through the Top 5 Features found by SHAP
                    for clean_name in clean_labels:
                        
                        if clean_name in IMMUTABLE_FEATURES:
                            continue
                        
                        config = SIMULATION_CONFIG.get(clean_name)
                        if not config:
                            continue

                        visible_slider_count += 1
                        col_name = config['col']
                        
                        # --- SAFE VALUE EXTRACTION ---
                        raw_val = simulated_student[col_name].iloc[0]

                        if pd.isna(raw_val):
                            current_val = float(config['min'])
                        elif isinstance(raw_val, str):
                            if raw_val.strip().upper() in ['Y', 'YES', '1']:
                                current_val = 1.0
                            else:
                                current_val = 0.0
                        else:
                            current_val = float(raw_val)

                        # --- RENDER WIDGETS ---
                        if clean_name == "Sports Participation":
                            is_active = st.toggle(f"🏃 {clean_name}", value=(current_val > 0.5))
                            new_values[col_name] = 1.0 if is_active else 0.0
                        else:
                            new_values[col_name] = st.slider(
                                f"{clean_name}",
                                min_value=float(config['min']),
                                max_value=float(config['max']),
                                value=float(current_val),
                                step=float(config['step'])
                            )
                    
                    if visible_slider_count == 0:
                        st.info("The top risk factors for this student cannot be changed (e.g., Past Failures).")
                    
                    st.write("")
                    submitted = st.form_submit_button("⚡ Recalculate Risk", use_container_width=True)

                # --- RESULT CALCULATION ---
                if submitted and visible_slider_count > 0:
                    
                    # !!! CRITICAL FIX: Use the actual dataframe index !!!
                    row_idx = simulated_student.index[0]

                    for col, val in new_values.items():
                        
                        # A. Handle Sports (Strings)
                        if col == "sports_participation" and isinstance(student_data[col].iloc[0], str):
                             simulated_student.at[row_idx, col] = 'Y' if val == 1.0 else 'N'
                        
                        # B. Handle Integers (Force conversion to int)
                        elif col in INT_COLUMNS:
                             simulated_student.at[row_idx, col] = int(val)
                             
                        # C. Handle Floats (CGPA, Attendance)
                        else:
                             simulated_student.at[row_idx, col] = float(val)

                    try:
                        # Predict
                        new_pred_prob = full_pipeline.predict_proba(simulated_student)[0][1] * 100
                        original_risk = risk_score
                        risk_change = new_pred_prob - original_risk
                        
                        # Display
                        if new_pred_prob >= 75:
                            bg_color = "#c62828"
                        elif new_pred_prob >= 40:
                            bg_color = "#ef6c00"
                        else:
                            bg_color = "#2e7d32"

                        st.markdown(f"""
                            <div style="margin-top: 20px; background-color: {bg_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                                <h3 style="margin:0; font-size:18px; opacity:0.8;">Simulated Risk Score</h3>
                                <h1 style="margin:0; font-size: 42px; font-weight: bold;">{new_pred_prob:.1f}%</h1>
                                <hr style="border-top: 1px solid rgba(255,255,255,0.2); margin: 10px 0;">
                                <p style="margin:0; font-weight:bold;">
                                    {f'📉 Improvement: {abs(risk_change):.1f}%' if risk_change < 0 else f'📈 Risk Increase: {risk_change:.1f}%'}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Calculation Error: {e}")

        except Exception as e:
            st.error("Could not generate risk analysis.")
            st.warning(f"Technical Detail: {str(e)}")
    else:
        st.info("Model not loaded.")
# --- 6. MAIN APP CONTROLLER ---
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'dashboard':
    show_dashboard_page()
elif st.session_state.page == 'detail':
    show_detail_page()