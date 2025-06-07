
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Custom CSS styling
st.markdown("""
    <style>
    /* Background and layout */
    .main {
        background-color: #f7fbff;
    }
    body {
        background-image: url("https://www.transparenttextures.com/patterns/white-wall-3.png");
        background-size: cover;
    }

    /* Button styles */
    .stDownloadButton button, .stButton button {
        background-color: #07072c !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 16px;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #1a1a5c !important;
    }

    /* Section headings */
    h1, h2, h3 {
        color: #07072c;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Divider style */
    .divider {
        border-top: 3px solid #07072c;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Result cards */
    .result-card {
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .survival {
        background-color: #28a745;
    }
    .death {
        background-color: #dc3545;
    }

    /* Sidebar */
    .css-1d391kg, .css-1v0mbdj {
        background-color: #f2f2ff;
        color: #07072c;
    }

    </style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('random_forest_model.joblib')
        num_imputer = joblib.load('num_imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        cat_imputer = joblib.load('cat_imputer.joblib')
        encoder = joblib.load('encoder.joblib')
        numeric_features = joblib.load('numeric_features.joblib')
        categorical_features = joblib.load('categorical_features.joblib')
        processed_columns = joblib.load('processed_columns.joblib')
        return model, num_imputer, scaler, cat_imputer, encoder, numeric_features, categorical_features, processed_columns
    except FileNotFoundError:
        st.error("Model or preprocessor files not found.")
        return None, None, None, None, None, None, None, None

model, num_imputer, scaler, cat_imputer, encoder, numeric_features, categorical_features, processed_columns = load_resources()

if model:
    st.sidebar.title("🏥 Hospital Death Predictor")
    st.sidebar.markdown("Upload patient data to predict ICU mortality outcomes.")

    st.title("🎯 Predict Patient Survival")

    st.markdown("### 📄 Download Sample CSV")
    sample_data = [
        {
            "age": 68.0, "bmi": 22.73, "elective_surgery": 0, "ethnicity": "Caucasian", "gender": "M", "height": 180.3,
            "icu_stay_type": "admit", "icu_type": "CTICU", "pre_icu_los_days": 0.541667, "weight": 73.9,
            "apache_2_diagnosis": 113.0, "apache_3j_diagnosis": 502.01, "arf_apache": 0,
            "gcs_eyes_apache": 3.0, "gcs_motor_apache": 6.0, "gcs_unable_apache": 0.0, "gcs_verbal_apache": 4.0,
            "heart_rate_apache": 118.0, "intubated_apache": 0.0, "map_apache": 40.0, "resprate_apache": 36.0,
            "temp_apache": 39.3, "ventilated_apache": 0.0, "d1_diasbp_max": 68.0, "d1_diasbp_min": 37.0,
            "d1_heartrate_max": 119.0, "d1_heartrate_min": 72.0, "d1_mbp_max": 89.0, "d1_mbp_min": 46.0,
            "d1_resprate_max": 34.0, "d1_resprate_min": 10.0, "d1_spo2_max": 100.0, "d1_spo2_min": 74.0,
            "d1_sysbp_max": 131.0, "d1_sysbp_min": 73.0, "d1_temp_max": 39.9, "d1_temp_min": 37.2,
            "h1_diasbp_max": 68.0, "h1_diasbp_min": 63.0, "h1_heartrate_max": 119.0, "h1_heartrate_min": 108.0,
            "h1_mbp_max": 86.0, "h1_mbp_min": 85.0, "h1_resprate_max": 26.0, "h1_resprate_min": 18.0,
            "h1_spo2_max": 100.0, "h1_spo2_min": 74.0, "h1_sysbp_max": 131.0, "h1_sysbp_min": 115.0,
            "d1_glucose_max": 168.0, "d1_glucose_min": 109.0, "d1_potassium_max": 4.0, "d1_potassium_min": 3.4, "aids": 0.0, "cirrhosis": 0.0,
            "diabetes_mellitus": 1.0, "hepatic_failure": 0.0, "immunosuppression": 0.0, "leukemia": 0.0,
            "lymphoma": 0.0, "solid_tumor_with_metastasis": 0.0
        }
    ]  # Your sample dictionary as-is
    sample_df = pd.DataFrame(sample_data)
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV", data=sample_csv, file_name="sample_input.csv", mime="text/csv")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📤 Upload Patient Data CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            missing_numeric = [col for col in numeric_features if col not in input_df.columns]
            missing_categorical = [col for col in categorical_features if col not in input_df.columns]
            if missing_numeric or missing_categorical:
                st.error(f"Missing required columns: {missing_numeric + missing_categorical}")
                st.stop()

            # Preprocess
            input_num_imputed = num_imputer.transform(input_df[numeric_features])
            input_num_scaled = scaler.transform(input_num_imputed)
            input_num_scaled_df = pd.DataFrame(input_num_scaled, columns=numeric_features)

            input_cat_imputed = cat_imputer.transform(input_df[categorical_features])
            input_cat_encoded = encoder.transform(input_cat_imputed)
            input_cat_encoded_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(categorical_features))

            input_processed = pd.concat([input_num_scaled_df, input_cat_encoded_df], axis=1)

            missing_cols = set(processed_columns) - set(input_processed.columns)
            for c in missing_cols:
                input_processed[c] = 0
            input_processed = input_processed[processed_columns]

            predictions = model.predict(input_processed)
            probabilities = model.predict_proba(input_processed)[:, 1]

            result_df = input_df.copy()
            result_df["Predicted Outcome"] = ["Death" if p == 1 else "Survival" for p in predictions]
            result_df["Death Probability"] = probabilities.round(4)
            result_df["Will Die (Yes/No)"] = ["Yes" if p == 1 else "No" for p in predictions]

            st.markdown("### ✅ Prediction Results")
            st.dataframe(result_df)

            # Large Card Result if only 1 patient
            if len(result_df) == 1:
                outcome = "death" if predictions[0] == 1 else "survival"
                st.markdown(f"""
                    <div class="result-card {'death' if outcome=='death' else 'survival'}">
                        <h2>🩺 Prediction: {outcome.capitalize()}</h2>
                        
                    </div>
                """, unsafe_allow_html=True)

            result_csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction Results", data=result_csv, file_name="hospital_death_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
