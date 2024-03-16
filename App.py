import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Load the trained XGBoost model
xg = joblib.load("xgboost.pkl")
df = pd.read_csv("bank.csv")

# Define a function to preprocess user input
def preprocess_input(data):
    # Encode categorical variables using LabelEncoder
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical variables using StandardScaler
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data

# Define the Streamlit app
def main():
    # Display title and image
    st.title("Bank Direct Marketing Prediction")
    st.image("image1.png", use_column_width=True)

    # Login system
    session_state = st.session_state
    if "authenticated" not in session_state:
        session_state.authenticated = False

    if not session_state.authenticated:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":
                session_state.authenticated = True
                st.sidebar.success("Logged in as Admin")
            else:
                st.sidebar.error("Incorrect username or password")
    else:
        st.sidebar.write("Logged in as Admin")
        if st.sidebar.button("Logout"):
            session_state.authenticated = False
            st.sidebar.info("Logged out")
            st.stop()  # Stop Streamlit app when logout button is clicked

        # Display main content when authenticated
        st.subheader("Sample Data")
        st.dataframe(df.tail())

        # Gather user input
        age = st.slider("Age", min_value=18, max_value=95, step=1)
        job = st.selectbox("Job Category", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
        education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Has Default?", ["yes", "no"])
        balance = st.number_input("Balance", value=-6847.0, min_value=-6847.0, max_value=81204.0)
        housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
        loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        day = st.slider("Day of Contact", min_value=1, max_value=31, step=1)
        month = st.selectbox("Month of Contact", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Duration of Contact", value=2.0, min_value=2.0, max_value=3881.0)
        campaign = st.slider("Number of Contacts During Campaign", min_value=1, max_value=63, step=1)
        pdays = st.slider("Number of Days Since Last Contact", min_value=-1, max_value=854, step=1)
        previous = st.slider("Number of Contacts Before This Campaign", min_value=0, max_value=58, step=1)
        poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "other", "success", "unknown"])

        # Create a DataFrame from the user input
        user_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'day': [day],
            'month': [month],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome]
        })

        # Define a flag to check if prediction has been made
        prediction_made = False

        # Check if the "Predict" button is clicked
        if st.button("Predict"):
            prediction_made = True
            # Preprocess the user input
            preprocessed_user_data = preprocess_input(user_data)

            # Make prediction using the XGBoost model
            prediction = xg.predict(preprocessed_user_data)

            # Display the prediction
            st.subheader("Prediction:")
            if prediction[0] == 1:
                st.success("Customer is likely to subscribe to a term deposit.")
            else:
                st.error("Customer is unlikely to subscribe to a term deposit.")

        # Display a message if prediction has not been made yet
        if not prediction_made:
            st.write("Click the 'Predict' button to see the prediction.")

        st.markdown("---")

        # Display visualizations
        st.subheader("Data Visualizations")
        
        # Distribution plot of age
        st.write("Distribution of Age")
        fig_age, ax_age = plt.subplots()
        sns.histplot(df['age'], ax=ax_age)
        st.pyplot(fig_age, clear_figure=True)
        st.markdown("---")

        # Distribution plot of balance
        st.write("Distribution of Balance")
        fig_balance, ax_balance = plt.subplots()
        sns.histplot(df['balance'], ax=ax_balance)
        st.pyplot(fig_balance, clear_figure=True)
        st.markdown("---")

        # Count plot for job category
        st.write("Count of Job Categories")
        fig_job, ax_job = plt.subplots()
        sns.countplot(x='job', data=df, ax=ax_job)
        st.pyplot(fig_job, clear_figure=True)
        st.markdown("---")


        # Distribution plot of duration
        st.write("Distribution of Duration")
        fig_duration, ax_duration = plt.subplots()
        sns.histplot(df['duration'], ax=ax_duration)
        st.pyplot(fig_duration, clear_figure=True)
        st.markdown("---")

        # Count plot for marital data
        st.write("Count of Marital Status")
        fig_marital, ax_marital = plt.subplots(figsize=(10, 6))
        sns.countplot(x=df['marital'], ax=ax_marital)
        st.pyplot(fig_marital, clear_figure=True)
        st.markdown("---")

        # Plot for campaign category
        st.write("Distribution of Campaign")
        fig_campaign, ax_campaign = plt.subplots(figsize=(15, 6))
        sns.histplot(df['campaign'], kde=False, ax=ax_campaign)
        st.pyplot(fig_campaign, clear_figure=True)
        st.markdown("---")

        # Plot for target variable
        st.write("Count of Subscription")
        fig_subscription, ax_subscription = plt.subplots(figsize=(10, 5))
        sns.countplot(x=df['deposit'], ax=ax_subscription)
        st.pyplot(fig_subscription, clear_figure=True)

if __name__ == "__main__":
    main()
