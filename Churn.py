import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---------------- SIDEBAR OPTION ---------------- #
option = st.sidebar.selectbox("Choose Section:", ("Single Input", "Bulk Upload", "Visual Analytics"))

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

# ---------------- VISUAL ANALYTICS ---------------- #
elif option == "Visual Analytics":
    st.header("📊 Customer Churn - Visual Analytics")
    file = st.file_uploader("Upload CSV for Visual Analytics", type=["csv"], key="viz_csv")

    if file is not None:
        df = pd.read_csv(file)

        # Clean numeric columns
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
        df['CLTV'] = pd.to_numeric(df['CLTV'], errors='coerce')
        df['Churn_Score'] = pd.to_numeric(df['Churn_Score'], errors='coerce')
        df.dropna(subset=['Total_Charges', 'CLTV', 'Churn_Score'], inplace=True)

        # --- Churn Risk Group ---
        bins = [0, 25, 50, 75, 100]
        labels = ['Low Risk (0-25)', 'Medium Risk (26-50)', 'High Risk (51-75)', 'Critical (76-100)']
        df['Churn Risk Group'] = pd.cut(df['Churn_Score'], bins=bins, labels=labels, include_lowest=True)
        risk_group_counts = df['Churn Risk Group'].value_counts().sort_index()

        st.subheader("Churn Score Bins")
        fig, ax = plt.subplots(figsize=(8, 5))
        risk_group_counts.plot(kind='bar', color='skyblue', ax=ax)
        for i, value in enumerate(risk_group_counts):
            ax.text(i, value + 2, str(value), ha='center', fontweight='bold')
        ax.set_title('Customer Distribution by Churn Score Bins')
        ax.set_xlabel('Churn Risk Group')
        ax.set_ylabel('Number of Customers')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # --- CLTV Group ---
        df['CLTV Group'] = pd.cut(df['CLTV'], bins=[0, 2000, 5000, df['CLTV'].max()],
                                  labels=['Low', 'Medium', 'High'], include_lowest=True)

        # --- Risk Matrix ---
        st.subheader("Churn Risk vs CLTV Group – Heatmap")
        risk_matrix = pd.crosstab(df['Churn Risk Group'], df['CLTV Group'])
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(risk_matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax2)
        ax2.set_title("Risk Matrix: Churn Risk vs CLTV Group")
        st.pyplot(fig2)

        # --- Total Charges Summary ---
        st.subheader("Total Charges Summary by Churn Label")
        summary_table = df.groupby('Churn_Label')['Total_Charges'].describe()[['count', 'min', '25%', '50%', '75%', 'max']].round(2)
        summary_table.rename(columns={'25%': 'Q1', '50%': 'Median', '75%': 'Q3'}, inplace=True)
        st.dataframe(summary_table)

        # --- Strip Plot of Total Charges ---
        st.subheader("Total Charges by Churn Label – Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.stripplot(x='Churn_Label', y='Total_Charges', data=df, jitter=0.3, palette='Set2', alpha=0.7, ax=ax3)
        ax3.set_title('Total Charges Distribution by Churn Status')
        st.pyplot(fig3)

        # --- CLTV vs Churn Score Scatter with Risk Zone ---
        st.subheader("CLTV vs Churn Score – Risk Segmentation")
        cltv_threshold = df['CLTV'].quantile(0.75)
        churn_score_threshold = 75
        df['Risk_Segment'] = df.apply(lambda row: 
            'Danger' if row['CLTV'] >= cltv_threshold and row['Churn_Score'] >= churn_score_threshold else 'Safe',
            axis=1
        )
        segment_summary = df['Risk_Segment'].value_counts()
        st.write("Risk Segment Summary:")
        st.dataframe(segment_summary)

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='CLTV', y='Churn_Score', hue='Churn_Label', palette='coolwarm', alpha=0.7, ax=ax4)
        ax4.axhline(churn_score_threshold, color='red', linestyle='--', linewidth=1, label='Churn Score = 75')
        ax4.axvline(cltv_threshold, color='orange', linestyle='--', linewidth=1, label='High CLTV (75th percentile)')
        ax4.set_title('CLTV vs Churn Score – High CLTV & High Churn Risk = Danger')
        ax4.legend()
        st.pyplot(fig4)
