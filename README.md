
# Employee Prediction App using Support Vector Machine

This project is a Streamlit-based web application that predicts whether an employee earns **above or below the median salary** using classification models. The best-performing model (based on F1 Score, Accuracy, and Recall) has been selected and saved for deployment.

---

## 📦 Project Structure

```
employee_salary_app/
│
├── app.py                       # Streamlit app
├── best_salary_model.pkl       # Trained SVC model, feature list, scaler, and selector
├── Salary_Data.csv             # Sample dataset
├── batch_salary_predictions.csv (optional output after upload)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd employee_salary_app
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

This will open the app in your browser at `http://localhost:8501`.

---

## 🖥️ How to Use the App

### 🔹 Individual Prediction

1. Use the **Sidebar** to input:
   - Age
   - Gender
   - Education Level
   - Job Title
   - Years of Experience

2. Click on **🔍 Predict** to see:
   - Whether the employee earns above or below the median salary.
   - A confidence score.
   - The selected profile in a table.

---

### 🔹 Batch Prediction (CSV Upload)

1. Prepare a CSV file with the following columns:
   ```
   Age, Gender, Education Level, Job Title, Years of Experience
   ```

2. Upload the file using the **📂 Batch Prediction** section.

3. The app will:
   - Perform predictions on all records.
   - Show a table of results with predictions and confidence.
   - Show a summary chart.
   - Provide a **📥 Download** button for results.

---

## 📊 Model Info

- **Model Used**: Support Vector Machine (SVC)
- **Selected Based On**: Accuracy, Recall, F1 Score
- **Features Used**:
  - Age
  - Years of Experience
  - Gender (encoded)
  - Education Level (encoded)
  - Job Title (one-hot encoded)

---

## 🧪 Sample Data

You can use the provided `Salary_Data.csv` or generate your own CSV using the expected format for testing the batch upload.

---

## 📎 Requirements

See `requirements.txt`, which includes:

```bash
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## 📞 Contact

For feedback, issues, or questions, please open an issue in the repository or contact the developer.

---
