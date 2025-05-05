import streamlit as st
import pandas as pd
import joblib

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# ---------------- LOGIN LOGIC ---------------- #
def login():
    st.title("🔐 Login to Prediction App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOGOUT BUTTON ---------------- #
st.title("📈 Customer Churn Prediction App")
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- Load Model ---------------- #
model = joblib.load('dt_model.pkl')

# Final feature list
features = ['Count', 'Latitude', 'Longitude', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Tenure_Months',
            'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup',
            'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies', 'Contract',
            'Paperless_Billing', 'Payment_Method', 'Monthly_Charges', 'Total_Charges', 'Churn_Value',
            'Churn_Score', 'CLTV']

# Define feature types
continuous_features = ['Latitude', 'Longitude', 'Monthly_Charges', 'Total_Charges', 'CLTV', 'Churn_Score', 'Tenure_Months', 'Count']
binary_features = ['Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service', 'Paperless_Billing']
multiclass_features = ['Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup',
                       'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies',
                       'Contract', 'Payment_Method']

# Sidebar option
option = st.sidebar.selectbox("Select Prediction Type:", ("Single Input", "Bulk Upload"))

# ---------------- SINGLE INPUT ---------------- #
if option == "Single Input":
    st.header("🧍‍♂️ Enter Customer Details:")
    input_data = {}

    for feature in features:
        if feature in continuous_features:
            input_data[feature] = st.number_input(f"{feature}:", value=0.0)
        elif feature in binary_features:
            input_data[feature] = st.selectbox(f"{feature}:", [0, 1])
        elif feature in multiclass_features:
            input_data[feature] = st.selectbox(f"{feature}:", [0, 1, 2])  # Assuming 3 classes
        elif feature == 'Churn_Value':
            input_data[feature] = st.selectbox(f"{feature}:", [0, 1])  # Included as feature

    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")

# ---------------- BULK UPLOAD ---------------- #
elif option == "Bulk Upload":
    st.header("📤 Upload CSV File:")
    file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        data = data[features]  # Ensure proper column order

        predictions = model.predict(data)
        data['Prediction'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]
        st.write(data)

        # Downloadable CSV
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )
