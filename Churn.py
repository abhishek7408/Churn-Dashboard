import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- User authentication ---
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
        else:
            st.error("Invalid credentials. Try again.")

# --- Logout button ---
def logout():
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Show login or dashboard ---
if not st.session_state.logged_in:
    login()
    st.stop()
else:
    logout()

# Page settings
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Analysis Dashboard")

# Load data
file_path = "Telecom Final Churn Sheet.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Sidebar filters
st.sidebar.header("🔎 Filter the Data")
gender_options = df['Gender'].dropna().unique()
contract_options = df['Contract'].dropna().unique()
churn_label_options = df['Churn_Label'].dropna().unique().tolist()

selected_gender = st.sidebar.multiselect("Select Gender(s):", gender_options, default=gender_options)
selected_contract = st.sidebar.multiselect("Select Contract Type(s):", contract_options, default=contract_options)
selected_churn_label = st.sidebar.multiselect("Select Churn Label(s):", churn_label_options, default=churn_label_options)

# Apply filters
filtered_df = df[
    (df['Gender'].isin(selected_gender)) &
    (df['Contract'].isin(selected_contract)) &
    (df['Churn_Label'].isin(selected_churn_label))
]

# KPIs
st.header("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_tenure = filtered_df['Tenure_Months'].mean()
    st.metric("Average Tenure (Months)", f"{avg_tenure:.1f}")

with col2:
    avg_monthly_charges = filtered_df['Monthly_Charges'].mean()
    st.metric("Avg Monthly Charges ($)", f"${avg_monthly_charges:.2f}")

with col3:
    churn_rate = (filtered_df['Churn_Value'].sum() / filtered_df.shape[0]) * 100 if filtered_df.shape[0] > 0 else 0
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}%")

with col4:
    avg_cltv = filtered_df['CLTV'].mean()
    st.metric("Avg Customer Lifetime Value", f"${avg_cltv:.0f}")

# Add Total Customers KPI
st.markdown("### 🧍 Total Customers: **{}**".format(len(filtered_df)))

st.markdown("---")

# Pie Chart - Churn Distribution
st.subheader("🔵 Churn Distribution")
fig_pie = px.pie(
    filtered_df,
    names="Churn_Value",
    title="Churn vs Non-Churn Customers",
    color_discrete_sequence=px.colors.qualitative.Set2,
    hole=0.4
)
fig_pie.update_traces(textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# Bar Chart - Contract vs Churn
st.subheader("📑 Contract Type and Churn")
fig_contract = px.histogram(
    filtered_df,
    x="Contract",
    color="Churn_Value",
    barmode="group",
    labels={"Churn_Value": "Churn (1=Yes, 0=No)"},
    title="Contract Types vs Churn"
)
st.plotly_chart(fig_contract, use_container_width=True)

st.markdown("---")

# Line Chart - Tenure vs Monthly Charges
st.subheader("📈 Tenure vs Monthly Charges")
fig_line = px.scatter(
    filtered_df,
    x="Tenure_Months",
    y="Monthly_Charges",
    color="Churn_Value",
    labels={"Churn_Value": "Churn (1=Yes, 0=No)"},
    title="Tenure vs Monthly Charges",
    trendline="ols"
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
numeric_features = filtered_df.select_dtypes(include=['float64', 'int64'])
corr = numeric_features.corr()
fig_corr, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig_corr)
