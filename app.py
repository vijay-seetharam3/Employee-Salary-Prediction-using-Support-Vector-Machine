import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns

# Load model, features, and scaler
model, feature_names, scaler, selector = joblib.load('best_salary_model.pkl')


# Sidebar and Header
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")
st.markdown("Predict whether an employee earns **above or below** the median salary.")

# Collect user input
st.sidebar.header("Employee Information")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.sidebar.selectbox("Job Title", [
    'Junior Marketing Coordinator', 'Senior Financial Manager', 'Senior Data Scientist',
    'Junior Sales Representative', 'Senior Marketing Specialist', 'Senior HR Specialist',
    'Junior Operations Manager', 'Senior Marketing Coordinator', 'Senior Data Engineer',
    'Junior Marketing Manager', 'Senior Financial Analyst', 'Director of Marketing',
    'Junior Business Analyst', 'Senior Project Manager', 'Senior Data Analyst',
    'Junior Financial Analyst', 'Senior Product Manager', 'Director of Operations',
    'Junior Operations Analyst', 'Senior Project Coordinator', 'Junior Business Development Associate',
    'Senior Product Designer', 'Senior Marketing Analyst', 'Senior Software Engineer',
    'Junior Product Manager', 'Senior Business Analyst', 'Junior Marketing Specialist',
    'Senior Business Development Manager', 'Senior HR Manager', 'Senior Financial Advisor',
    'Junior HR Coordinator', 'Junior Financial Advisor', 'Senior UX Designer',
    'Director of Engineering', 'Junior Operations Coordinator', 'Director of HR'
])
Experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Prepare raw input
input_df = pd.DataFrame({
    'Age': [age],
    'Years of Experience': [Experience],
    'Gender_Male': [1 if gender == 'Male' else 0],
    "Education Level_Master's": [1 if education == "Master's" else 0],
    'Education Level_PhD': [1 if education == "PhD" else 0],
})


# One-hot encode job title
job_cols = [col for col in feature_names if col.startswith('Job Title')]
job_encoded = {col: [0] for col in job_cols}
selected_job_col = f'Job Title_{job_title}'
if selected_job_col in job_encoded:
    job_encoded[selected_job_col] = [1]
job_df = pd.DataFrame(job_encoded)

# Combine all features
X_input = pd.concat([input_df, job_df], axis=1)

# Add missing columns
for col in feature_names:
    if col not in X_input.columns:
        X_input[col] = 0

# Reorder and scale
X_input = X_input[feature_names]
X_input = pd.DataFrame(scaler.transform(X_input), columns=feature_names)

# Display selected profile
st.subheader("📋 Selected Employee Profile")
st.dataframe(pd.DataFrame({
    "Feature": ["Age", "Gender", "Education", "Job Title", "Years of Experience"],
    "Value": list(map(str, [age, gender, education, job_title, Experience]))
}))

import matplotlib.pyplot as plt

# Prediction button
if st.button("🔍 Predict"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.success(f"✅ Predicted: Employee earns **above** the median salary ({proba*100:.1f}% confidence).")
    else:
        st.warning(f"❌ Predicted: Employee earns **below** the median salary ({(1-proba)*100:.1f}% confidence).")

    st.markdown("---")
    st.caption("Model based on historical employee dataset. Results are indicative.")



st.markdown("---")
st.header("📂 Batch Prediction: Upload Employee Dataset")

uploaded_file = st.file_uploader("Upload a CSV file with employee data", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)

        expected_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        if not all(col in df_upload.columns for col in expected_cols):
            st.error(f"❌ Uploaded file must contain these columns: {expected_cols}")
        else:
            df_batch = df_upload.copy()

            # Basic feature engineering
            df_batch['Gender_Male'] = df_batch['Gender'].apply(lambda x: 1 if x.strip().lower() == 'male' else 0)
            df_batch["Education Level_Master's"] = df_batch['Education Level'].apply(lambda x: 1 if x.strip() == "Master's" else 0)
            df_batch["Education Level_PhD"] = df_batch['Education Level'].apply(lambda x: 1 if x.strip() == "PhD" else 0)

            # One-hot encoding for Job Title
            for col in feature_names:
                if col.startswith('Job Title'):
                    df_batch[col] = df_batch['Job Title'].apply(lambda x: 1 if f"Job Title_{x.strip()}" == col else 0)

            # Add missing columns with 0s
            for col in feature_names:
                if col not in df_batch.columns:
                    df_batch[col] = 0

            # Create input in right order
            X_batch = df_batch[feature_names]

            X_batch = pd.DataFrame(scaler.transform(X_batch), columns=feature_names)


            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1]

            df_upload['Prediction'] = np.where(preds == 1, 'Above Median', 'Below Median')
            df_upload['Confidence (%)'] = (np.maximum(probs, 1 - probs) * 100).round(1)

            # Show results
            st.subheader("📊 Prediction Results")
            st.dataframe(df_upload)

            st.subheader("📈 Prediction Summary")
            fig, ax = plt.subplots()
            sns.countplot(data=df_upload, x='Prediction', hue='Prediction', palette='Set2', legend=False)
            st.pyplot(fig)

            # Download button
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="batch_salary_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error processing uploaded file: {e}")
