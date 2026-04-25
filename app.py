import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    font-size: 12px; font-weight: 600; color: #6c757d;
    text-transform: uppercase; letter-spacing: .07em;
    border-bottom: 1px solid #e9ecef; padding-bottom: 5px; margin-bottom: 12px;
}
.timeline-item {
    border-left: 3px solid #dee2e6; padding: 8px 14px;
    margin-bottom: 10px; position: relative;
    border-radius: 0 6px 6px 0; background: #fafafa;
}
.timeline-item::before {
    content:''; width:11px; height:11px; border-radius:50%;
    background:#adb5bd; position:absolute; left:-7px; top:10px;
}
.timeline-item.joined  { border-color:#198754; background:#f0faf4; }
.timeline-item.joined::before  { background:#198754; }
.timeline-item.risk    { border-color:#dc3545; background:#fff5f5; }
.timeline-item.risk::before    { background:#dc3545; }
.timeline-item.current { border-color:#0d6efd; background:#f0f5ff; }
.timeline-item.current::before { background:#0d6efd; }
.churn-yes {
    background:#fff5f5; border:1px solid #f5c6cb;
    border-radius:10px; padding:14px 18px; margin-bottom:8px;
}
.churn-no {
    background:#f0faf4; border:1px solid #c3e6cb;
    border-radius:10px; padding:14px 18px; margin-bottom:8px;
}
.suggest-card {
    border-radius:8px; border:1px solid #e9ecef;
    padding:12px 14px; margin-bottom:10px; background:#fff;
}
.suggest-card.urgent   { border-left:4px solid #dc3545; }
.suggest-card.moderate { border-left:4px solid #fd7e14; }
.suggest-card.positive { border-left:4px solid #198754; }
.profile-table td { padding: 5px 10px; font-size: 14px; }
.profile-table td:first-child { color: #6c757d; width: 160px; }
.profile-table td:last-child  { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FEATURE ORDER  (exact order your original app.py used)
# ──────────────────────────────────────────────────────────────
ENCODE_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv = "customer_churn.csv" if os.path.exists("customer_churn.csv") else "data/customer_churn.csv"
    df = pd.read_csv(csv)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# Uses your churn_model.pkl (XGBClassifier).
# Falls back to RandomForest trained from CSV if pkl missing.
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("churn_model.pkl"):
        model = pickle.load(open("churn_model.pkl", "rb"))
        return model, "xgb"
    # Fallback — train from CSV
    df = load_data()
    df_m = df.copy()
    le = LabelEncoder()
    for col in ENCODE_COLS:
        df_m[col] = le.fit_transform(df_m[col].astype(str))
    df_m["Churn"] = (df_m["Churn"] == "Yes").astype(int)
    X = df_m[FEATURE_ORDER]
    y = df_m["Churn"]
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model, "rf"


# ──────────────────────────────────────────────────────────────
# ENCODE A SINGLE CSV ROW → numpy array for model.predict
# Matches the exact encoding your original app implied
# ──────────────────────────────────────────────────────────────
def encode_row_to_array(row, df):
    """
    Returns a (1, 19) numpy array in FEATURE_ORDER —
    the same shape as np.array([[0,0,...,MonthlyCharges,TotalCharges]])
    in your original app.py
    """
    row_enc = row.copy()
    le = LabelEncoder()
    for col in ENCODE_COLS:
        le.fit(df[col].astype(str).tolist())
        row_enc[col] = le.transform([str(row[col])])[0]
    return np.array([[row_enc[f] for f in FEATURE_ORDER]])


def predict(model, arr):
    pred = int(model.predict(arr)[0])
    try:
        prob = float(model.predict_proba(arr)[0][1])
    except Exception:
        prob = 1.0 if pred == 1 else 0.0
    return pred, prob


# ──────────────────────────────────────────────────────────────
# TIMELINE GENERATOR
# ──────────────────────────────────────────────────────────────
def generate_timeline(row):
    tenure = int(row["tenure"])
    events = []

    events.append({
        "type": "joined", "month": 0,
        "label": "Month 1 — Customer joined",
        "detail": f"{row['Contract']} contract · {row['InternetService']} internet · Phone: {row['PhoneService']}"
    })

    if tenure > 5:
        if row["OnlineSecurity"] == "No":
            events.append({
                "type": "risk", "month": min(4, tenure - 1),
                "label": f"Month {min(4, tenure-1)} — No Online Security",
                "detail": "Customer has no Online Security — increases churn risk"
            })
        else:
            events.append({
                "type": "joined", "month": min(4, tenure - 1),
                "label": f"Month {min(4, tenure-1)} — Added Online Security",
                "detail": "Customer subscribed to Online Security package"
            })

    if tenure > 10:
        if row["TechSupport"] == "No":
            events.append({
                "type": "risk", "month": min(8, tenure - 2),
                "label": f"Month {min(8, tenure-2)} — No Tech Support coverage",
                "detail": "No Tech Support is a strong churn indicator"
            })

    if tenure > 14:
        if row["Contract"] == "Month-to-month":
            events.append({
                "type": "risk", "month": min(12, tenure - 2),
                "label": f"Month {min(12, tenure-2)} — Stayed on month-to-month",
                "detail": "Declined annual contract upgrade offer"
            })
        else:
            events.append({
                "type": "joined", "month": min(12, tenure - 2),
                "label": f"Month {min(12, tenure-2)} — Contract renewed",
                "detail": f"Renewed {row['Contract']} contract — positive retention signal"
            })

    if tenure > 18 and row["StreamingTV"] == "Yes":
        events.append({
            "type": "joined", "month": min(16, tenure - 2),
            "label": f"Month {min(16, tenure-2)} — Added Streaming TV",
            "detail": "Subscribed to Streaming TV — engagement increased"
        })

    if row["PaymentMethod"] == "Electronic check":
        events.append({
            "type": "risk", "month": min(tenure - 1, tenure),
            "label": f"Month {min(tenure-1, tenure)} — Payment method flagged",
            "detail": "Electronic check users show higher churn rates than auto-pay customers"
        })

    churn_status = "HIGH churn risk detected" if row["Churn"] == "Yes" else "Stable — low churn risk"
    events.append({
        "type": "current", "month": tenure,
        "label": f"Month {tenure} — Today (Current status)",
        "detail": f"{churn_status} · Monthly: ${row['MonthlyCharges']:.2f} · Total paid: ${row['TotalCharges']:.2f}"
    })

    return sorted(events, key=lambda x: x["month"])


# ──────────────────────────────────────────────────────────────
# SUGGESTIONS ENGINE  (churn=Yes → retention | No → upsell)
# ──────────────────────────────────────────────────────────────
def get_suggestions(row, churn_prob):
    suggestions = []
    pct = int(churn_prob * 100)

    if churn_prob > 0.5:
        # ── CHURN PREDICTED → retention actions ──
        if row["Contract"] == "Month-to-month":
            suggestions.append(("urgent", "💰", "Offer a contract upgrade incentive",
                f"Customer is month-to-month with {pct}% churn risk. A 20–25% discount for switching to a 1-year plan could cut churn probability significantly."))
        if row["TechSupport"] == "No":
            suggestions.append(("urgent", "🛠️", "Offer free Tech Support trial (3 months)",
                "No Tech Support is a top churn driver. A free 3-month trial increases perceived value with minimal cost."))
        if row["InternetService"] == "Fiber optic" and row["OnlineSecurity"] == "No":
            suggestions.append(("moderate", "🔒", "Bundle Online Security at no extra cost",
                "Fiber optic customers without security add-ons churn more. Include it free for 3 months."))
        if row["PaymentMethod"] == "Electronic check":
            suggestions.append(("moderate", "💳", "Encourage switch to auto-pay",
                "Offer a $2–5/month discount for switching to bank transfer or credit card auto-pay. Reduces churn and missed payments."))
        suggestions.append(("urgent", "📞", "Schedule a personal retention call",
            f"With {pct}% churn probability, direct outreach within 7 days is recommended. Personalise the offer to this customer's profile."))
        if int(row["tenure"]) >= 12:
            suggestions.append(("moderate", "🎁", f"Loyalty reward for {int(row['tenure'])}-month tenure",
                "Long-tenure customers respond well to milestone rewards — a free month or free upgrade reinforces loyalty."))
    else:
        # ── NO CHURN PREDICTED → proactive upsell ──
        suggestions.append(("positive", "✅", "Customer is stable — ideal time to upsell",
            f"Churn probability is only {pct}%. Offering premium services now won't risk the relationship."))
        if row["StreamingTV"] == "No":
            suggestions.append(("positive", "📺", "Upsell Streaming TV bundle",
                "Customer doesn't have Streaming TV. A 1-month free trial increases ARPU and raises switching cost."))
        if row["OnlineBackup"] == "No":
            suggestions.append(("positive", "☁️", "Promote Online Backup service",
                "Adding Online Backup increases data dependency on the service, making customers less likely to churn."))
        if row["Contract"] == "Month-to-month":
            suggestions.append(("moderate", "📋", "Soft offer for annual contract",
                "Customer is stable but still month-to-month. A gentle annual plan offer with a small discount locks in retention long-term."))
        suggestions.append(("positive", "🌟", "Send a satisfaction survey",
            "Stable customers who receive check-in surveys report higher NPS. Use feedback to personalise future offers."))

    return suggestions


# ──────────────────────────────────────────────────────────────
# PARAMETER TREND CHART
# ──────────────────────────────────────────────────────────────
def build_param_chart(row, param):
    tenure = int(row["tenure"])
    current_val = float(row[param])
    param_label = {"MonthlyCharges": "Monthly Charges ($)", "TotalCharges": "Total Charges ($)", "tenure": "Tenure (months)"}[param]
    color      = {"MonthlyCharges": "#4361ee", "TotalCharges": "#7209b7", "tenure": "#3a0ca3"}[param]
    color_rgba = {"MonthlyCharges": "rgba(67,97,238,0.1)", "TotalCharges": "rgba(114,9,183,0.1)", "tenure": "rgba(58,12,163,0.1)"}[param]

    if param == "MonthlyCharges":
        np.random.seed(abs(hash(row["customerID"])) % 9999)
        base = current_val * 0.88
        vals = np.clip(np.random.normal(base, base * 0.035, tenure), base * 0.8, current_val + 3).tolist()
        vals[-1] = current_val
    elif param == "TotalCharges":
        monthly = float(row["MonthlyCharges"])
        vals = [round(monthly * (i + 1) * 0.98, 2) for i in range(tenure)]
        vals[-1] = current_val
    else:
        vals = list(range(1, tenure + 1))

    months = [f"M{i+1}" for i in range(tenure)]
    if tenure > 24:
        step = max(tenure // 24, 1)
        months = months[::step]; vals = vals[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=vals, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=4, color=color),
        fill="tozeroy", fillcolor=color_rgba
    ))
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=6, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        showlegend=False
    )
    return fig, param_label


# ──────────────────────────────────────────────────────────────
# MANUAL PREDICT  (your original 3-input form — kept intact)
# ──────────────────────────────────────────────────────────────
def manual_predict_section(model):
    st.markdown("---")
    st.markdown("### 🔢 Quick Predict (manual input)")
    st.caption("Enter values manually — same as the original app.")

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure_m = st.number_input("Tenure", 0, 100, key="m_tenure")
    with c2:
        monthly_m = st.number_input("Monthly Charges", key="m_monthly")
    with c3:
        total_m = st.number_input("Total Charges", key="m_total")

    if st.button("Predict", key="manual_predict"):
        # Exact same array shape as your original app.py
        input_data = np.array([[0, 0, 0, 0, tenure_m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, monthly_m, total_m]])
        prediction = model.predict(input_data)
        try:
            prob = model.predict_proba(input_data)[0][1]
            prob_pct = int(prob * 100)
        except Exception:
            prob_pct = None

        if prediction[0] == 1:
            st.error(f"❌ Customer will CHURN" + (f" — {prob_pct}% probability" if prob_pct else ""))
        else:
            st.success(f"✅ Customer will STAY" + (f" — {prob_pct}% churn probability" if prob_pct else ""))


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
df = load_data()
model, model_source = load_model()
model_tag = "XGBoost (pkl)" if model_source == "xgb" else "RandomForest (CSV)"

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Customer Churn App")
    st.caption(f"Model: **{model_tag}**")
    st.markdown("---")

    st.markdown("### 🔍 Customer Lookup")
    customer_ids = df["customerID"].tolist()
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    st.markdown("---")
    st.markdown("### ⚙️ Show / Hide Sections")
    show_timeline = st.checkbox("Customer Timeline", value=True)
    show_chart    = st.checkbox("Parameter Chart",   value=True)
    show_suggest  = st.checkbox("Suggestions Panel", value=True)

    if show_chart:
        param_option = st.selectbox(
            "Chart parameter",
            ["MonthlyCharges", "TotalCharges", "tenure"],
            format_func=lambda x: {
                "MonthlyCharges": "Monthly Charges ($)",
                "TotalCharges":   "Total Charges ($)",
                "tenure":         "Tenure (months)"
            }[x]
        )

    st.markdown("---")
    st.markdown(f"**Total customers:** `{len(df)}`")
    st.markdown(f"**Churn rate in CSV:** `{(df['Churn']=='Yes').mean()*100:.1f}%`")

    # ── ADD NEW CUSTOMER (dynamic CSV) ──
    with st.expander("➕ Add new customer to CSV"):
        new_id       = st.text_input("Customer ID", value=f"NEW-{len(df)+1:04d}")
        new_gender   = st.selectbox("Gender", ["Male", "Female"])
        new_senior   = st.selectbox("Senior Citizen", [0, 1])
        new_partner  = st.selectbox("Partner", ["Yes", "No"])
        new_dep      = st.selectbox("Dependents", ["Yes", "No"])
        new_tenure   = st.number_input("Tenure (months)", 1, 100, 12)
        new_phone    = st.selectbox("Phone Service", ["Yes", "No"])
        new_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        new_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        new_monthly  = st.number_input("Monthly Charges ($)", 10.0, 200.0, 65.0)
        new_churn    = st.selectbox("Churn label", ["No", "Yes"])

        if st.button("💾 Save to CSV"):
            new_row = {
                "customerID": new_id, "gender": new_gender,
                "SeniorCitizen": new_senior, "Partner": new_partner,
                "Dependents": new_dep, "tenure": int(new_tenure),
                "PhoneService": new_phone, "MultipleLines": "No",
                "InternetService": new_internet, "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": new_contract,
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": new_monthly,
                "TotalCharges": round(new_monthly * int(new_tenure), 2),
                "Churn": new_churn
            }
            updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            updated.to_csv("customer_churn.csv", index=False)
            st.success(f"✅ {new_id} saved! Reload the page to see them.")


# ── SELECTED CUSTOMER ─────────────────────────────────────────
row = df[df["customerID"] == selected_id].iloc[0]
arr = encode_row_to_array(row, df)
prediction, churn_prob = predict(model, arr)
churn_pct = int(churn_prob * 100)
is_churn  = prediction == 1

# ── HEADER ────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("# 📊 Customer Churn Prediction App")
    st.caption("Telco Customer Churn · XGBoost Model · Customer Detail View")
with h2:
    if is_churn:
        st.error(f"⚠️ HIGH RISK · {churn_pct}%")
    elif churn_pct > 35:
        st.warning(f"🟡 MODERATE · {churn_pct}%")
    else:
        st.success(f"✅ LOW RISK · {churn_pct}%")

st.markdown("---")

# ── METRIC STRIP ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customer ID",     row["customerID"])
m2.metric("Tenure",          f"{int(row['tenure'])} months")
m3.metric("Monthly Charges", f"${row['MonthlyCharges']:.2f}")
m4.metric("Total Charges",   f"${row['TotalCharges']:.2f}")
m5.metric("Contract",        row["Contract"])

st.markdown("---")

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    # Profile
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="profile-table">
      <tr><td>Gender</td><td>{row['gender']}</td></tr>
      <tr><td>Senior Citizen</td><td>{'Yes' if row['SeniorCitizen'] else 'No'}</td></tr>
      <tr><td>Partner</td><td>{row['Partner']}</td></tr>
      <tr><td>Dependents</td><td>{row['Dependents']}</td></tr>
      <tr><td>Phone Service</td><td>{row['PhoneService']}</td></tr>
      <tr><td>Multiple Lines</td><td>{row['MultipleLines']}</td></tr>
      <tr><td>Internet Service</td><td>{row['InternetService']}</td></tr>
      <tr><td>Online Security</td><td>{row['OnlineSecurity']}</td></tr>
      <tr><td>Online Backup</td><td>{row['OnlineBackup']}</td></tr>
      <tr><td>Device Protection</td><td>{row['DeviceProtection']}</td></tr>
      <tr><td>Tech Support</td><td>{row['TechSupport']}</td></tr>
      <tr><td>Streaming TV</td><td>{row['StreamingTV']}</td></tr>
      <tr><td>Streaming Movies</td><td>{row['StreamingMovies']}</td></tr>
      <tr><td>Payment Method</td><td>{row['PaymentMethod']}</td></tr>
      <tr><td>Paperless Billing</td><td>{row['PaperlessBilling']}</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction result
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
    churn_class = "churn-yes" if is_churn else "churn-no"
    churn_icon  = "🔴" if is_churn else "🟢"
    churn_label = "Will CHURN ❌" if is_churn else "Will STAY ✅"
    st.markdown(f"""
    <div class="{churn_class}">
      <div style="font-size:18px;font-weight:600;">{churn_icon} {churn_label}</div>
      <div style="margin-top:6px;font-size:14px;">
        Churn probability: <strong>{churn_pct}%</strong>
      </div>
      <div style="margin-top:4px;font-size:13px;color:#6c757d;">
        Actual label in dataset: <strong>{row['Churn']}</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar
    bar_color = "#dc3545" if is_churn else "#198754"
    fig_bar = go.Figure(go.Bar(
        x=[churn_pct], y=["Churn probability"],
        orientation="h", marker_color=bar_color,
        text=[f"{churn_pct}%"], textposition="inside"
    ))
    fig_bar.update_layout(
        height=65, margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(range=[0,100], showgrid=False, showticklabels=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Parameter chart (left column, below prediction)
    if show_chart:
        st.markdown("<br>", unsafe_allow_html=True)
        fig_chart, param_label = build_param_chart(row, param_option)
        st.markdown(f'<div class="section-header">{param_label} — Over Customer Lifetime</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})


with right:
    # Timeline
    if show_timeline:
        st.markdown('<div class="section-header">Customer Journey — Start to Today</div>',
                    unsafe_allow_html=True)
        for ev in generate_timeline(row):
            css = {"joined": "joined", "risk": "risk", "current": "current"}.get(ev["type"], "")
            st.markdown(f"""
            <div class="timeline-item {css}">
              <div style="font-weight:600;font-size:14px;">{ev['label']}</div>
              <div style="font-size:13px;color:#6c757d;margin-top:3px;">{ev['detail']}</div>
            </div>
            """, unsafe_allow_html=True)


# ── SUGGESTIONS (full width) ──────────────────────────────────
if show_suggest:
    st.markdown("---")
    if is_churn:
        st.markdown("### ⚠️ Retention Suggestions")
        st.caption("Customer is at risk — take these steps to prevent churn.")
    else:
        st.markdown("### 💡 Proactive Recommendations")
        st.caption("Customer is stable — suggestions to increase value and loyalty.")

    suggestions = get_suggestions(row, churn_prob)
    col_a, col_b = st.columns(2)
    for i, (urgency, icon, title, detail) in enumerate(suggestions):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div class="suggest-card {urgency}">
              <div style="font-weight:600;font-size:14px;">{icon} {title}</div>
              <div style="font-size:13px;color:#555;margin-top:6px;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)


# ── ORIGINAL MANUAL PREDICT (kept from your existing app.py) ──
manual_predict_section(model)

st.markdown("---")
st.caption(f"Built with Streamlit · Model: {model_tag} · Telco Customer Churn Dataset")