# Heart Disease Risk Prediction System

A comprehensive machine learning system for predicting heart disease risk using patient medical data, now powered by **Streamlit** for an enhanced interactive experience.

## 📋 Project Overview

This project uses the Cleveland Heart Disease dataset to train machine learning models that can predict the likelihood of heart disease based on 13 medical parameters. The application has been upgraded from a Flask web app to a modern Streamlit dashboard with advanced reporting features.

### 🎯 Key Features
- **Interactive Streamlit Dashboard**: Clean, professional, and responsive user interface.
- **Real-time Predictions**: Instant risk assessment using a trained Random Forest model.
- **Health Parameter Analysis**: Detailed breakdown of medical parameters (Normal, Borderline, Abnormal).
- **Risk Factor Identification**: Highlights the major contributing factors to the predicted risk.
- **Hospital-Style Reports**: Generates professional PDF reports with:
    - Patient Demographics
    - Vital Signs & Cardiac Assessment
    - Color-coded Risk Level
    - Medical Recommendations
    - Report Preview & Download
- **History Tracking**: Automatically saves and displays past predictions.
- **High Accuracy**: Random Forest model achieves **88.52% accuracy**.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the Dashboard
The application will automatically open in your default browser at:
`http://localhost:8501`

## 📊 Model Performance

| Model | Accuracy | AUC Score | Status |
|-------|----------|-----------|---------|
| **Random Forest** | **88.52%** | **0.954** | 🏆 Best |
| Logistic Regression | 86.89% | 0.953 | ✅ Good |
| Gradient Boosting | 85.25% | 0.946 | ✅ Good |
| SVM | 67.21% | 0.794 | ⚠️ Fair |

## 📁 Project Structure

```
HEART_DISEASE/
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training script
├── HEART_cleveland.csv             # Dataset
├── prediction_history.csv          # Saved predictions history
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── training_summary.md             # Detailed training report
```

## 🔧 Technical Details

### Dataset
- **Source**: Cleveland Clinic Foundation
- **Features**: 13 medical parameters
- **Target**: Binary classification (0=No Disease, 1=Disease)

### Medical Parameters Analyzed
1. `Age`: Patient age in years
2. `Sex`: Biological sex
3. `Chest Pain Type (cp)`: Typical angina, Atypical angina, Non-anginal, Asymptomatic
4. `Resting BP (trestbps)`: Resting blood pressure (mm Hg)
5. `Cholesterol (chol)`: Serum cholesterol (mg/dl)
6. `Fasting Blood Sugar (fbs)`: > 120 mg/dl indicator
7. `Resting ECG (restecg)`: Wave abnormalities or hypertrophy
8. `Max Heart Rate (thalach)`: Maximum heart rate achieved
9. `Exercise Angina (exang)`: Exercise induced angina
10. `ST Depression (oldpeak)`: ST depression induced by exercise
11. `ST Slope (slope)`: Slope of the peak exercise ST segment
12. `Major Vessels (ca)`: Number of major vessels colored by fluoroscopy
13. `Thalassemia (thal)`: Blood disorder status

### Risk Classification
- 🟢 **Low Risk (<30%)**: Normal range, maintain healthy lifestyle.
- 🟡 **Medium Risk (30-70%)**: Borderline risk, medical consultation advised.
- 🔴 **High Risk (>70%)**: Critical risk, immediate medical attention recommended.

## 🌐 Application Features

### 🔮 Prediction Page
- **Patient Form**: Easy-to-use form for entering medical data.
- **Instant Analysis**: Immediate calculation of risk probability.
- **Visual Feedback**: Success/Warning/Error indicators based on risk level.
- **Downloadable PDF**: Professional medical report generation with preview.

### 📜 History Page
- **Records**: View table of all past predictions.
- **Export**: Download history as CSV for external analysis.

## ⚠️ Medical Disclaimer

This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 🔄 Recent Updates
- **v2.0 (Streamlit Migration)**:
    - Replaced Flask backend with Streamlit.
    - Added comprehensive Health Parameter Analysis.
    - Implemented Feature Importance for risk factor explanation.
    - Added professional PDF report generation with hospital branding.
    - Removed emojis for a clinical, professional appearance.

---
**Model**: Random Forest (n_estimators=100) | **Accuracy**: 88.52% | **Updated**: Jan 2026
