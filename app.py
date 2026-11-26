from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Telco Churn & Revenue-at-Risk Dashboard",
    layout="wide",
)


@st.cache_data
def load_data():
  data_path = Path(__file__).parent / "data" / "scored_customers.csv"
  return pd.read_csv(data_path)


@st.cache_resource
def load_model_bundle():
  model_path = Path(__file__).parent / "data" / "churn_model.pkl"
  return joblib.load(model_path)


def reset_filters():
  st.session_state["contract_filter"] = "All"
  st.session_state["internet_filter"] = "All"
  st.session_state["tech_support_filter"] = "All"


df = load_data()
bundle = load_model_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_columns"]

X_base = pd.get_dummies(
    df.drop(columns=["Churn", "Churn_Probability", "Revenue_at_Risk"]),
    drop_first=True,
    dtype=int,
).reindex(columns=feature_cols, fill_value=0)
baseline_row = X_base.mean().to_frame().T

st.sidebar.header("Dashboard Filters")

contract_label_map = {
    "Month-to-month": "Month-to-Month",
    "One year": "One Year",
    "Two year": "Two Year",
}
reverse_contract_map = {v: k for k, v in contract_label_map.items()}

internet_label_map = {
    "Fiber optic": "Fiber Optic",
    "DSL": "DSL",
    "No": "No Internet Service",
    "No internet service": "No Internet Service",
}
reverse_internet_map = {v: k for k, v in internet_label_map.items()}

tech_support_label_map = {
    "1": "Has Tech Support",
    "Yes": "Has Tech Support",
    "0": "No Tech Support",
    "No": "No Tech Support",
    "No internet service": "No Internet Service",
    "No Internet Service": "No Internet Service",
}
reverse_tech_support_map = {
    "Has Tech Support": ["1", "Yes"],
    "No Tech Support": ["0", "No"],
    "No Internet Service": ["No internet service", "No Internet Service"],
}

raw_contract_vals = df["Contract"].dropna().unique().tolist()
contract_display_options = ["All"] + sorted(
    {contract_label_map.get(val, str(val)) for val in raw_contract_vals}
)
selected_contract_label = st.sidebar.selectbox(
    "Contract Type", contract_display_options, key="contract_filter"
)

raw_internet_vals = df["InternetService"].dropna().unique().tolist()
internet_display_options = ["All"] + sorted(
    {internet_label_map.get(val, str(val)) for val in raw_internet_vals}
)
selected_internet_label = st.sidebar.selectbox(
    "Internet Service", internet_display_options, key="internet_filter"
)

raw_tech_vals = (
    df["TechSupport"].dropna().astype(str).str.strip().unique().tolist()
)
tech_display_options = ["All"] + sorted(
    {tech_support_label_map.get(val, val) for val in raw_tech_vals}
)
selected_tech_label = st.sidebar.selectbox(
    "Tech Support", tech_display_options, key="tech_support_filter"
)

st.sidebar.button(
    "Clear Filters", on_click=reset_filters, use_container_width=True
)

filtered_df = df.copy()
if selected_contract_label != "All":
  target_contract = reverse_contract_map.get(
      selected_contract_label, selected_contract_label
  )
  filtered_df = filtered_df[
      (filtered_df["Contract"] == target_contract)
      | (filtered_df["Contract"] == selected_contract_label)
  ]
if selected_internet_label != "All":
  target_internet = reverse_internet_map.get(
      selected_internet_label, selected_internet_label
  )
  filtered_df = filtered_df[
      (filtered_df["InternetService"] == target_internet)
      | (filtered_df["InternetService"] == selected_internet_label)
  ]
if selected_tech_label != "All":
  valid_tech_vals = reverse_tech_support_map.get(
      selected_tech_label, [selected_tech_label]
  )
  filtered_df = filtered_df[
      filtered_df["TechSupport"].astype(str).str.strip().isin(valid_tech_vals)
  ]

st.sidebar.markdown("---")
st.sidebar.header("What-If Churn Predictor")

sim_tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=13)
sim_monthly = st.sidebar.slider(
    "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=79.0, step=0.5
)
sim_contract_label = st.sidebar.selectbox(
    "Simulated Contract", ["Month-to-Month", "One Year", "Two Year"]
)
sim_internet_label = st.sidebar.selectbox(
    "Simulated Internet", ["Fiber Optic", "DSL", "No Internet Service"]
)

sim_contract_raw = reverse_contract_map.get(
    sim_contract_label, sim_contract_label
)
sim_internet_raw = reverse_internet_map.get(
    sim_internet_label, sim_internet_label
)

sim_input = baseline_row.copy()
if "tenure" in sim_input.columns:
  sim_input["tenure"] = sim_tenure
if "MonthlyCharges" in sim_input.columns:
  sim_input["MonthlyCharges"] = sim_monthly
if "TotalCharges" in sim_input.columns:
  sim_input["TotalCharges"] = sim_tenure * sim_monthly

for col in [c for c in sim_input.columns if c.startswith("Contract_")]:
  sim_input[col] = 0
if f"Contract_{sim_contract_raw}" in sim_input.columns:
  sim_input[f"Contract_{sim_contract_raw}"] = 1

for col in [c for c in sim_input.columns if c.startswith("InternetService_")]:
  sim_input[col] = 0
if f"InternetService_{sim_internet_raw}" in sim_input.columns:
  sim_input[f"InternetService_{sim_internet_raw}"] = 1

sim_scaled = scaler.transform(sim_input[feature_cols])
sim_prob = model.predict_proba(sim_scaled)[0, 1] * 100

st.sidebar.metric("Predicted Churn Risk", f"{sim_prob:.2f}%")
if sim_prob >= 70:
  st.sidebar.error("High Risk — Immediate Retention Offer Recommended")
elif sim_prob >= 40:
  st.sidebar.warning("Moderate Risk — Monitor & Engage")
else:
  st.sidebar.success("Low Risk — Stable Customer Profile")

st.title("Customer Churn & Revenue-at-Risk Dashboard")

total_customers = len(filtered_df)
overall_churn_rate = (
    filtered_df["Churn"].mean() * 100 if total_customers > 0 else 0.0
)
monthly_rev_at_risk = filtered_df["Revenue_at_Risk"].sum()

high_risk_df = filtered_df[filtered_df["Churn_Probability"] > 70]
potential_annual_savings = (high_risk_df["MonthlyCharges"].sum() * 12) * 0.30

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Active Customers", f"{total_customers:,}")
kpi2.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%")
kpi3.metric("Monthly Revenue at Risk", f"${monthly_rev_at_risk:,.2f}")
kpi4.metric(
    "Potential Annual Savings (30% Saved)", f"${potential_annual_savings:,.2f}"
)

st.markdown("---")

st.subheader("Churn Drivers: Average Churn Risk by Contract & Tech Support")

chart_df = filtered_df.copy()
chart_df["Contract_Label"] = chart_df["Contract"].replace(contract_label_map)
chart_df["TechSupport_Label"] = (
    chart_df["TechSupport"]
    .astype(str)
    .str.strip()
    .replace({
        "1": "Has Tech Support",
        "Yes": "Has Tech Support",
        "0": "No Tech Support",
        "No": "No Tech Support",
        "No internet service": "No Internet Service",
        "No Internet Service": "No Internet Service",
    })
)

grouped_drivers = (
    chart_df.groupby(["Contract_Label", "TechSupport_Label"], as_index=False)[
        "Churn_Probability"
    ]
    .mean()
    .sort_values("Churn_Probability", ascending=True)
)

fig = px.bar(
    grouped_drivers,
    x="Churn_Probability",
    y="Contract_Label",
    color="TechSupport_Label",
    barmode="group",
    orientation="h",
    labels={
        "Churn_Probability": "Average Churn Probability (%)",
        "Contract_Label": "Contract Type",
        "TechSupport_Label": "Tech Support",
    },
    text_auto=".2f",
)
fig.update_layout(height=400, xaxis_range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("High-Risk Customer Watchlist (Churn Probability > 70%)")


def suggest_action(row):
  if row["Contract"] == "Month-to-month":
    return "Offer 1-year contract upgrade (15% discount)"
  if str(row["TechSupport"]).strip() in ["0", "No"]:
    return "Bundle free Tech Support for 3 months"
  return "Assign dedicated retention specialist call"


watchlist = (
    filtered_df[filtered_df["Churn_Probability"] > 70]
    .sort_values("MonthlyCharges", ascending=False)
    .copy()
)

if watchlist.empty:
  st.info("No customers exceed 70% churn probability under the current filters.")
else:
  watchlist["Suggested_Action"] = watchlist.apply(suggest_action, axis=1)
  display_cols = [
      "Contract",
      "InternetService",
      "tenure",
      "MonthlyCharges",
      "Churn_Probability",
      "Revenue_at_Risk",
      "Suggested_Action",
  ]
  st.dataframe(
      watchlist[display_cols].style.format({
          "MonthlyCharges": "${:,.2f}",
          "Churn_Probability": "{:.1f}%",
          "Revenue_at_Risk": "${:,.2f}",
      }),
      use_container_width=True,
  )