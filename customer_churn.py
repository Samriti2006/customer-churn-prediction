import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load dataset
df = pd.read_csv('data/customer_churn.csv')

# 2️⃣ Quick look
print("First 5 rows:\n", df.head())
print("\nInfo:\n", df.info())

# 3️⃣ Handle missing or blank TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 4️⃣ Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    if col != 'customerID':  # ID column skip
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 5️⃣ Split features & target
X = df.drop(['customerID','Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 7️⃣ Predict & evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Show dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())
# Visualization

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
# Example prediction

sample_customer = X.iloc[[0]]
prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("Prediction: Customer will Churn")
else:
    print("Prediction: Customer will Stay")
    # -------- Customer Prediction Example --------

print("\nCustomer Churn Prediction Example")

sample_customer = X.iloc[[0]]   # first customer data

prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("Prediction: Customer will CHURN (leave)")
else:
    print("Prediction: Customer will STAY")
    import joblib

# Save the trained model
joblib.dump(model, "churn_model.pkl")

print("Model saved successfully!")
import pickle

# Load saved model
with open('churn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("\nModel loaded successfully!")

# Example input for prediction
example_customer = X_test.iloc[0].values.reshape(1,-1)

prediction = loaded_model.predict(example_customer)

if prediction[0] == 1:
    print("Prediction: Customer will CHURN")
else:
    print("Prediction: Customer will STAY")
    import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Churn Count
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()


# Graph 2: Monthly Charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()


# Graph 3: Contract Type vs Churn
plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.show()