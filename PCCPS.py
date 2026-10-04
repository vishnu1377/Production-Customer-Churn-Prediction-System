import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import streamlit as st

df = pd.read_csv("PCCPS.csv")

st.title('Production Customer Churn Prediction System')
st.subheader('Dataset Overview')
st.write(df.head())
st.write('Dataset Information:')
st.write(df.info())
st.write(df.describe())
st.write(df.isnull().sum())

st.write(df.columns.tolist())
print(df.columns.tolist())

df.drop(['customerID'], axis=1, inplace=True)

categorical_columns = df.select_dtypes(include='object').columns
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

plt.hist(df['tenure'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.title('Distribution of Tenure')
plt.show()
plt.gcf().clear()

df['Churn'] = df['Churn'].map({0: 'No Churn', 1: 'Churn'})
st.write(df['Churn'].value_counts())

st.write('Time series features:')
df['MonthlyCharges_lag1'] = df['MonthlyCharges'].shift(1)
st.write(df[['MonthlyCharges', 'MonthlyCharges_lag1']].head())

st.write('Rolling features:')
df["MonthlyCharges_roll3"] = (df["MonthlyCharges"].rolling(3).mean())
st.write(df[['MonthlyCharges', 'MonthlyCharges_roll3']].head())

X = df.drop('Churn', axis=1)
y = df['Churn']

st.write('Columns with object dtype in the feature set:')
st.write(X.select_dtypes(include='object').columns.tolist())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write('Training and Testing Data Shapes:')
st.write(f'Training Data Shape: {x_train.shape}')
st.write(f'Testing Data Shape: {x_test.shape}')

st.write('Model Training and Evaluation:')
model = RandomForestClassifier(random_state=42)

st.write('confusion_matrix:')
st.write(confusion_matrix(y_test, model.fit(x_train, y_train).predict(x_test)))

st.write('classification_report:')
st.write(classification_report(y_test, model.predict(x_test)))

st.write('Performing Cross-Validation...')
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')
st.write('Cross-Validation Scores:')
st.write(scores)
st.write('Mean Cross-Validation Score:')
st.write(np.mean(scores))
st.write('Training the model on the entire training set...')
model.fit(x_train, y_train)

st.write('Performing Grid Search for Hyperparameter Tuning...')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

model = grid_search.best_estimator_ 
y_pred = model.predict(x_test)
st.write('Model Accuracy:')
st.write(accuracy_score(y_test, y_pred))

st.write('Enter the details of the customer to check for churn:')
Gender = st.selectbox("Select Gender", ['Male', 'Female'])
Senior_citizen = st.selectbox("Is Senior Citizen?", ['Yes', 'No'])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
Dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.number_input("Enter tenure (in months)")
phone_service = st.selectbox("Has Phone Service?", ['Yes', 'No'])
MultipleLines = st.selectbox("Has Multiple Lines?", ['Yes', 'No'])
InternetService = st.selectbox("Select Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Has Online Security?", ['Yes', 'No'])
OnlineBackup = st.selectbox("Has Online Backup?", ['Yes', 'No'])
DeviceProtection = st.selectbox("Has Device Protection?", ['Yes', 'No'])
TechSupport = st.selectbox("Has Tech Support?", ['Yes', 'No'])
StreamingTV = st.selectbox("Has Streaming TV?", ['Yes', 'No'])
StreamingMovies = st.selectbox("Has Streaming Movies?", ['Yes', 'No'])
Contract = st.selectbox("Select Contract Type", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Has Paperless Billing?", ['Yes', 'No'])
PaymentMethod = st.selectbox("Select Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Enter Monthly Charges")
TotalCharges = st.number_input("Enter Total Charges")

st.write('Predicting Churn...')
input_data = [Gender, Senior_citizen, partner, Dependents, tenure, phone_service, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
input_data = pd.DataFrame([input_data], columns=X.columns)
st.write('Input Data:')
st.write(input_data)
st.write('Prediction Result:')
prediction = model.predict(input_data)
st.write('Prediction value:', prediction[0])