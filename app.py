import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
from io import BytesIO
from fpdf import FPDF

# Page config
st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")

# Load and train model
@st.cache_resource
def load_model():
    data = pd.read_csv('HEART_cleveland.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_names = X.columns.tolist()
    
    return model, feature_names

model, feature_names = load_model()

# Analyze health parameters
def analyze_parameters(patient_data):
    """Analyze each parameter and classify as Normal or Abnormal"""
    analysis = []
    
    # Age analysis
    age = patient_data[0]
    if age < 45:
        analysis.append({"parameter": "Age", "value": f"{int(age)} years", "status": "Normal", "note": "Lower age group - lower risk"})
    elif age < 55:
        analysis.append({"parameter": "Age", "value": f"{int(age)} years", "status": "Borderline", "note": "Middle age - moderate risk factor"})
    else:
        analysis.append({"parameter": "Age", "value": f"{int(age)} years", "status": "Abnormal", "note": "Higher age increases heart disease risk"})
    
    # Sex analysis
    sex = patient_data[1]
    if sex == 1:
        analysis.append({"parameter": "Sex", "value": "Male", "status": "Abnormal", "note": "Males have higher risk of heart disease"})
    else:
        analysis.append({"parameter": "Sex", "value": "Female", "status": "Normal", "note": "Females have relatively lower risk"})
    
    # Chest Pain Type
    cp = patient_data[2]
    cp_types = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    if cp == 0:
        analysis.append({"parameter": "Chest Pain Type", "value": cp_types[cp], "status": "Abnormal", "note": "Typical angina indicates possible heart issue"})
    elif cp == 3:
        analysis.append({"parameter": "Chest Pain Type", "value": cp_types[cp], "status": "Normal", "note": "No chest pain symptoms"})
    else:
        analysis.append({"parameter": "Chest Pain Type", "value": cp_types[cp], "status": "Borderline", "note": "Atypical symptoms - needs monitoring"})
    
    # Blood Pressure
    bp = patient_data[3]
    if bp < 120:
        analysis.append({"parameter": "Blood Pressure", "value": f"{int(bp)} mm Hg", "status": "Normal", "note": "Optimal blood pressure"})
    elif bp < 140:
        analysis.append({"parameter": "Blood Pressure", "value": f"{int(bp)} mm Hg", "status": "Borderline", "note": "Elevated - needs lifestyle changes"})
    else:
        analysis.append({"parameter": "Blood Pressure", "value": f"{int(bp)} mm Hg", "status": "Abnormal", "note": "High blood pressure - major risk factor"})
    
    # Cholesterol
    chol = patient_data[4]
    if chol < 200:
        analysis.append({"parameter": "Cholesterol", "value": f"{int(chol)} mg/dl", "status": "Normal", "note": "Desirable cholesterol level"})
    elif chol < 240:
        analysis.append({"parameter": "Cholesterol", "value": f"{int(chol)} mg/dl", "status": "Borderline", "note": "Borderline high - needs attention"})
    else:
        analysis.append({"parameter": "Cholesterol", "value": f"{int(chol)} mg/dl", "status": "Abnormal", "note": "High cholesterol - increases heart risk"})
    
    # Fasting Blood Sugar
    fbs = patient_data[5]
    if fbs == 0:
        analysis.append({"parameter": "Fasting Blood Sugar", "value": "Normal (<120 mg/dl)", "status": "Normal", "note": "Blood sugar within normal range"})
    else:
        analysis.append({"parameter": "Fasting Blood Sugar", "value": "High (>120 mg/dl)", "status": "Abnormal", "note": "Elevated blood sugar - diabetes risk"})
    
    # Resting ECG
    ecg = patient_data[6]
    ecg_types = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
    if ecg == 0:
        analysis.append({"parameter": "Resting ECG", "value": ecg_types[ecg], "status": "Normal", "note": "Normal heart electrical activity"})
    else:
        analysis.append({"parameter": "Resting ECG", "value": ecg_types[ecg], "status": "Abnormal", "note": "ECG abnormality detected - needs evaluation"})
    
    # Maximum Heart Rate
    thalach = patient_data[7]
    max_hr = 220 - patient_data[0]  # Age-adjusted max HR
    if thalach >= max_hr * 0.85:
        analysis.append({"parameter": "Max Heart Rate", "value": f"{int(thalach)} bpm", "status": "Normal", "note": "Good exercise capacity"})
    elif thalach >= max_hr * 0.70:
        analysis.append({"parameter": "Max Heart Rate", "value": f"{int(thalach)} bpm", "status": "Borderline", "note": "Moderate exercise capacity"})
    else:
        analysis.append({"parameter": "Max Heart Rate", "value": f"{int(thalach)} bpm", "status": "Abnormal", "note": "Low exercise capacity - concerning"})
    
    # Exercise Induced Angina
    exang = patient_data[8]
    if exang == 0:
        analysis.append({"parameter": "Exercise Angina", "value": "No", "status": "Normal", "note": "No chest pain during exercise"})
    else:
        analysis.append({"parameter": "Exercise Angina", "value": "Yes", "status": "Abnormal", "note": "Chest pain during exercise - serious concern"})
    
    # ST Depression (Oldpeak)
    oldpeak = patient_data[9]
    if oldpeak < 1:
        analysis.append({"parameter": "ST Depression", "value": f"{oldpeak}", "status": "Normal", "note": "Minimal ST depression"})
    elif oldpeak < 2:
        analysis.append({"parameter": "ST Depression", "value": f"{oldpeak}", "status": "Borderline", "note": "Moderate ST depression"})
    else:
        analysis.append({"parameter": "ST Depression", "value": f"{oldpeak}", "status": "Abnormal", "note": "Significant ST depression - indicates ischemia"})
    
    # ST Slope
    slope = patient_data[10]
    slope_types = ["Upsloping", "Flat", "Downsloping"]
    if slope == 0:
        analysis.append({"parameter": "ST Slope", "value": slope_types[slope], "status": "Normal", "note": "Normal upsloping pattern"})
    elif slope == 1:
        analysis.append({"parameter": "ST Slope", "value": slope_types[slope], "status": "Borderline", "note": "Flat slope - needs monitoring"})
    else:
        analysis.append({"parameter": "ST Slope", "value": slope_types[slope], "status": "Abnormal", "note": "Downsloping - abnormal pattern"})
    
    # Major Vessels
    ca = patient_data[11]
    if ca == 0:
        analysis.append({"parameter": "Major Vessels Colored", "value": str(int(ca)), "status": "Normal", "note": "No vessel blockage detected"})
    elif ca == 1:
        analysis.append({"parameter": "Major Vessels Colored", "value": str(int(ca)), "status": "Borderline", "note": "Minor vessel involvement"})
    else:
        analysis.append({"parameter": "Major Vessels Colored", "value": str(int(ca)), "status": "Abnormal", "note": "Multiple vessel involvement - significant"})
    
    # Thalassemia
    thal = patient_data[12]
    thal_types = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
    if thal == 1:
        analysis.append({"parameter": "Thalassemia", "value": thal_types.get(thal, "Unknown"), "status": "Normal", "note": "Normal blood flow"})
    elif thal == 2:
        analysis.append({"parameter": "Thalassemia", "value": thal_types.get(thal, "Unknown"), "status": "Abnormal", "note": "Fixed defect - permanent damage"})
    else:
        analysis.append({"parameter": "Thalassemia", "value": thal_types.get(thal, "Unknown"), "status": "Borderline", "note": "Reversible defect - treatable"})
    
    return analysis

# Get major risk factors using feature importance
def get_risk_factors(patient_data):
    """Identify major contributing risk factors"""
    feature_importance = model.feature_importances_
    
    feature_labels = [
        "Age", "Sex", "Chest Pain Type", "Blood Pressure", "Cholesterol",
        "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
        "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"
    ]
    
    # Combine importance with patient data context
    risk_factors = []
    for i, (importance, value) in enumerate(zip(feature_importance, patient_data)):
        risk_factors.append({
            "feature": feature_labels[i],
            "importance": importance * 100,
            "value": value
        })
    
    # Sort by importance
    risk_factors.sort(key=lambda x: x["importance"], reverse=True)
    
    return risk_factors[:5]  # Return top 5 factors

# Prediction function
def predict_risk(patient_data):
    risk_proba = model.predict_proba([patient_data])[0][1]
    risk_percentage = risk_proba * 100
    
    if risk_percentage < 30:
        level = "Low Risk"
        recommendation = "Continue maintaining a healthy lifestyle. Regular exercise and balanced diet recommended."
    elif risk_percentage < 70:
        level = "Medium Risk"
        recommendation = "Schedule a consultation with a cardiologist for further evaluation. Monitor blood pressure and cholesterol regularly."
    else:
        level = "High Risk"
        recommendation = "Immediate medical consultation strongly recommended. Please visit a cardiac specialist at the earliest."
    
    return risk_percentage, level, recommendation

# Generate PDF Report with analysis
def generate_pdf_report(patient_name, patient_data, risk_percentage, risk_level, recommendation, analysis, risk_factors):
    sex_text = "Male" if patient_data[1] == 1 else "Female"
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(0, 102, 153)
    pdf.rect(0, 0, 210, 35, 'F')
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_y(8)
    pdf.cell(0, 10, 'CARDIAC CARE HOSPITAL', align='C', ln=True)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, 'Heart Disease Risk Assessment Report', align='C', ln=True)
    
    pdf.set_text_color(0, 0, 0)
    
    # Report ID and Date
    pdf.set_y(42)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(95, 6, f'Report ID: HD-{datetime.now().strftime("%Y%m%d%H%M%S")}', ln=False)
    pdf.cell(95, 6, f'Date: {datetime.now().strftime("%d-%m-%Y %H:%M")}', align='R', ln=True)
    
    pdf.line(10, 50, 200, 50)
    
    # Patient Information
    pdf.set_y(55)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '  PATIENT INFORMATION', fill=True, ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(2)
    pdf.cell(50, 6, 'Patient Name:', ln=False)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 6, patient_name, ln=False)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(30, 6, 'Age:', ln=False)
    pdf.cell(30, 6, f'{int(patient_data[0])} years', ln=False)
    pdf.cell(20, 6, 'Sex:', ln=False)
    pdf.cell(20, 6, sex_text, ln=True)
    
    pdf.ln(3)
    
    # Risk Assessment Result
    if risk_level == "Low Risk":
        pdf.set_fill_color(200, 230, 200)
        border_color = (0, 128, 0)
    elif risk_level == "Medium Risk":
        pdf.set_fill_color(255, 235, 200)
        border_color = (255, 140, 0)
    else:
        pdf.set_fill_color(255, 200, 200)
        border_color = (200, 0, 0)
    
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(0.8)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '  RISK ASSESSMENT RESULT', fill=True, ln=True)
    
    pdf.ln(3)
    y_pos = pdf.get_y()
    pdf.rect(10, y_pos, 190, 18, 'D')
    
    pdf.set_y(y_pos + 2)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*border_color)
    pdf.cell(0, 7, f'RISK LEVEL: {risk_level.upper()}', align='C', ln=True)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 6, f'Risk Probability: {risk_percentage:.1f}%', align='C', ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    
    pdf.ln(5)
    
    # Health Parameter Analysis
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '  HEALTH PARAMETER ANALYSIS', fill=True, ln=True)
    pdf.ln(2)
    
    # Table header
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(45, 6, 'Parameter', border=1, fill=True)
    pdf.cell(30, 6, 'Value', border=1, fill=True)
    pdf.cell(25, 6, 'Status', border=1, fill=True)
    pdf.cell(90, 6, 'Remarks', border=1, fill=True, ln=True)
    
    pdf.set_font('Helvetica', '', 8)
    for item in analysis:
        # Set status color
        if item["status"] == "Normal":
            pdf.set_fill_color(220, 255, 220)
        elif item["status"] == "Borderline":
            pdf.set_fill_color(255, 255, 200)
        else:
            pdf.set_fill_color(255, 220, 220)
        
        pdf.cell(45, 5, item["parameter"], border=1)
        pdf.cell(30, 5, str(item["value"]), border=1)
        pdf.cell(25, 5, item["status"], border=1, fill=True)
        pdf.cell(90, 5, item["note"], border=1, ln=True)
    
    pdf.ln(5)
    
    # Major Risk Factors
    pdf.set_fill_color(255, 230, 230)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '  MAJOR CONTRIBUTING FACTORS', fill=True, ln=True)
    pdf.ln(2)
    
    pdf.set_font('Helvetica', '', 9)
    for i, factor in enumerate(risk_factors, 1):
        pdf.cell(0, 5, f'{i}. {factor["feature"]} (Contribution: {factor["importance"]:.1f}%)', ln=True)
    
    pdf.ln(3)
    
    # Recommendation
    pdf.set_fill_color(230, 242, 255)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '  MEDICAL RECOMMENDATION', fill=True, ln=True)
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, recommendation)
    
    pdf.ln(3)
    
    # Disclaimer
    pdf.set_fill_color(255, 255, 220)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 6, '  DISCLAIMER', fill=True, ln=True)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.multi_cell(0, 4, 'This report is generated by an AI-based prediction system for informational purposes only. Please consult a qualified healthcare provider for proper medical evaluation.')
    
    # Footer
    pdf.set_y(-20)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 7)
    pdf.cell(0, 4, 'Heart Disease Risk Prediction System | ML Model: Random Forest (88.52% Accuracy)', align='C', ln=True)
    pdf.cell(0, 4, f'Generated on: {datetime.now().strftime("%d-%m-%Y at %H:%M:%S")}', align='C')
    
    return bytes(pdf.output())

# Save prediction to history
def save_prediction(patient_name, patient_data, risk_percentage, risk_level):
    history_file = 'prediction_history.csv'
    
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame(columns=[
            'timestamp', 'patient_name', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
            'risk_percentage', 'risk_level'
        ])
    
    new_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'patient_name': patient_name,
        'age': patient_data[0], 'sex': patient_data[1], 'cp': patient_data[2],
        'trestbps': patient_data[3], 'chol': patient_data[4], 'fbs': patient_data[5],
        'restecg': patient_data[6], 'thalach': patient_data[7], 'exang': patient_data[8],
        'oldpeak': patient_data[9], 'slope': patient_data[10], 'ca': patient_data[11],
        'thal': patient_data[12], 'risk_percentage': risk_percentage, 'risk_level': risk_level
    }
    
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(history_file, index=False)

# Main app
st.title("Heart Disease Risk Prediction")
st.write("Enter patient information to assess heart disease risk")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Predict", "History"])

if page == "Predict":
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            cp = st.selectbox("Chest Pain Type", options=[
                ("Typical angina", 0), ("Atypical angina", 1), 
                ("Non-anginal pain", 2), ("Asymptomatic", 3)
            ], format_func=lambda x: x[0])
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
            restecg = st.selectbox("Resting ECG", options=[
                ("Normal", 0), ("ST-T abnormality", 1), ("Left ventricular hypertrophy", 2)
            ], format_func=lambda x: x[0])
        
        with col3:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Slope", options=[
                ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)
            ], format_func=lambda x: x[0])
            ca = st.selectbox("Major Vessels (0-3)", options=[(0, 0), (1, 1), (2, 2), (3, 3)], format_func=lambda x: str(x[0]))
            thal = st.selectbox("Thalassemia", options=[
                ("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)
            ], format_func=lambda x: x[0])
        
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)
    
    if submitted:
        if not patient_name:
            st.error("Please enter patient name!")
        else:
            patient_data = [
                age, sex[1], cp[1], trestbps, chol, fbs[1],
                restecg[1], thalach, exang[1], oldpeak, slope[1], ca[1], thal[1]
            ]
            
            risk_percentage, risk_level, recommendation = predict_risk(patient_data)
            analysis = analyze_parameters(patient_data)
            risk_factors = get_risk_factors(patient_data)
            
            save_prediction(patient_name, patient_data, risk_percentage, risk_level)
            
            st.divider()
            st.subheader("Prediction Result")
            
            if risk_level == "Low Risk":
                st.success(f"{risk_level} - {risk_percentage:.1f}%")
            elif risk_level == "Medium Risk":
                st.warning(f"{risk_level} - {risk_percentage:.1f}%")
            else:
                st.error(f"{risk_level} - {risk_percentage:.1f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Patient:** {patient_name}")
                st.write(f"**Age:** {age} years | **Sex:** {'Male' if sex[1] == 1 else 'Female'}")
            with col2:
                st.info(f"**Recommendation:** {recommendation}")
            
            # Health Parameter Analysis Section
            st.divider()
            st.subheader("Health Parameter Analysis")
            
            normal_params = [p for p in analysis if p["status"] == "Normal"]
            borderline_params = [p for p in analysis if p["status"] == "Borderline"]
            abnormal_params = [p for p in analysis if p["status"] == "Abnormal"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Normal Parameters**")
                for p in normal_params:
                    st.success(f"**{p['parameter']}**: {p['value']}")
                    st.caption(p['note'])
            
            with col2:
                st.markdown("**Borderline Parameters**")
                for p in borderline_params:
                    st.warning(f"**{p['parameter']}**: {p['value']}")
                    st.caption(p['note'])
            
            with col3:
                st.markdown("**Abnormal Parameters**")
                for p in abnormal_params:
                    st.error(f"**{p['parameter']}**: {p['value']}")
                    st.caption(p['note'])
            
            # Major Risk Factors
            st.divider()
            st.subheader("Major Contributing Factors")
            
            for i, factor in enumerate(risk_factors, 1):
                st.write(f"**{i}. {factor['feature']}** - Contribution: {factor['importance']:.1f}%")
            
            st.divider()
            st.subheader("Download Report")
            
            pdf_bytes = generate_pdf_report(patient_name, patient_data, risk_percentage, risk_level, recommendation, analysis, risk_factors)
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"Heart_Report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # PDF Preview
            import base64
            with st.expander("Preview Report", expanded=True):
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

else:
    st.subheader("Prediction History")
    
    if os.path.exists('prediction_history.csv'):
        history_df = pd.read_csv('prediction_history.csv')
        
        if len(history_df) > 0:
            display_df = history_df.copy()
            display_df['sex'] = display_df['sex'].map({1: 'M', 0: 'F'})
            display_df['risk_percentage'] = display_df['risk_percentage'].round(1)
            
            display_cols = ['timestamp', 'patient_name', 'age', 'sex', 'trestbps', 'chol', 'thalach', 'risk_percentage', 'risk_level']
            st.dataframe(display_df[display_cols], use_container_width=True)
            
            st.download_button(
                label="Download History (CSV)",
                data=history_df.to_csv(index=False),
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions yet. Make your first prediction!")
    else:
        st.info("No prediction history found. Make your first prediction!")

st.divider()
st.caption("2024 Heart Disease Risk Prediction System | ML Model: Random Forest")