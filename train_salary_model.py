# Employee Salary Prediction Pipeline (Custom Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy import stats

# Load Data
df = pd.read_csv('Salary_Data.csv')

# Data Cleaning
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df.dropna(inplace=True)

# Binary Classification - above median salary or not
df['Above_Median_Salary'] = (df['Salary'] > df['Salary'].median()).astype(int)

# EDA - Visualizations
sns.histplot(df['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

sns.heatmap(df[['Age', 'Years of Experience', 'Salary']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

numeric_cols = ['Age', 'Years of Experience', 'Salary']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col} (Outlier Detection)")
    plt.show()

# Outlier Removal using z-score
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

# Feature Engineering
categorical_cols = ['Gender', 'Education Level', 'Job Title']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Selection - Variance Threshold
X = df.drop(['Salary', 'Above_Median_Salary'], axis=1)
y = df['Above_Median_Salary']
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
selected_columns = X.columns[selector.get_support()]

# Convert X_selected back to DataFrame so scaler sees feature names
X = pd.DataFrame(X_selected, columns=selected_columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()

# Fit and transform training data with column names
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_columns)

# Transform test data and keep column names
X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_columns)


# Model Comparison and Saving Best Model
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True)
}

best_model = None
best_score = -np.inf
model_metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    model_metrics[name] = {'Accuracy': acc, 'Recall': rec, 'F1 Score': f1}
    print(f"{name}:")
    print(f"  Accuracy: {acc:.2f}")
    print(f"  Recall: {rec:.2f}")
    print(f"  F1 Score: {f1:.2f}\n")

    if f1 > best_score:
        best_score = f1
        best_model = model

# Visualization of Metrics
metrics_df = pd.DataFrame(model_metrics).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison - Accuracy, Recall, F1 Score")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the best model
# Save model, selected columns, scaler, and selector
joblib.dump((best_model, selected_columns.tolist(), scaler, selector), 'best_salary_model.pkl')
print("Best model saved as 'best_salary_model.pkl'")
