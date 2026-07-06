# 📊 Customer Churn Prediction

An interactive Streamlit web app that predicts whether a telecom customer is
likely to churn, using a machine learning model trained on the Telco Customer
Churn dataset.

## 🔗 Live Demo


## ✨ Features
- **Customer selector** — pick any customer by ID and view their full profile
- **Churn prediction** — instant risk classification (Low / Moderate / High)
  with a live probability bar
- **Customer journey timeline** — auto-generated month-by-month history
  highlighting risk events (e.g. no online security, no tech support,
  month-to-month contract)
- **Retention suggestions** — tailored recommendations depending on whether
  the customer is at risk or stable
- **Manual prediction form** — enter a hypothetical customer's details and
  get an instant churn prediction
- **Add new customer** — append a new customer record directly to the
  dataset from the UI
- **Interactive charts** — Plotly visualizations for monthly charges, total
  charges, and tenure trends

## 🛠️ Tech Stack
- **Frontend/App:** Streamlit
- **ML Model:** XGBoost Classifier (with a RandomForest fallback if the
  saved model isn't found)
- **Data Processing:** Pandas, NumPy, scikit-learn (Label Encoding)
- **Visualization:** Plotly

## 📁 Project Structure
```
customer-churn-prediction/
├── app.py                  # Streamlit app (main interface)
├── customer_churn.py       # Model training script
├── churn_model.pkl         # Saved trained XGBoost model
├── data/
│   └── customer_churn.csv  # Telco Customer Churn dataset
├── requirements.txt
└── README.md
```

## 📈 Model
The model is trained on the **Telco Customer Churn** dataset using an
`XGBClassifier`, with categorical features label-encoded and `TotalCharges`
cleaned of missing/blank values. Training and evaluation code (accuracy,
classification report, and exploratory visualizations) is in
`customer_churn.py`.

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Samriti2006/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 👩‍💻 Author
**Samriti**
📧 samritichoudhary682@gmail.com
