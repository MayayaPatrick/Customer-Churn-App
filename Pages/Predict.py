import streamlit as st
import joblib
import pandas as pd
import numpy as np

import streamlit as st


st.set_page_config(
    page_title='Predict',
    page_icon='',
    layout='wide'
)

# Load models using joblib
gradient_model = joblib.load(r"models\gradient_descent_model")
random_forest = joblib.load(r"models\random_forest")

def save_to_history(inputs, predicted_outcome, selected_model):
    # Load or create history dataframe in session state
    if 'history_df' not in st.session_state:
        st.session_state.history_df = pd.DataFrame(columns=['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService',
                                                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'PredictedOutcome'])

    # Append inputs and predicted outcome to history dataframe
    inputs['PredictedOutcome'] = predicted_outcome
    inputs['SelectedModel'] = selected_model
    st.session_state.history_df = pd.concat([st.session_state.history_df, inputs], ignore_index=True)

def customer_details():
    st.title('Customer Churn prediction page')
    st.write('Make your predictions here👇')

    model_choice_column, _ = st.columns([1, 3])  # Adjust the ratio as needed
    with model_choice_column:
        model_choice = st.selectbox('Select Model', ('Gradient Boosting', 'Random Forest'))

    with st.form('Customer_information'):
        # Section 1: Personal Details
        st.header('Personal Details')
        col1, col2 = st.columns(2)
        with col1:
            customer_ID = st.number_input('CustomerID:')
            gender = st.selectbox('Gender:', options=['Male', 'Female'])
            senior_citizen = st.selectbox('SeniorCitizen:', options=['Yes', 'No'])
            partner = st.selectbox('Partner:', options=['Yes', 'No'])
            dependents = st.selectbox('Dependents:', options=['Yes', 'No'])
            tenure = st.number_input('Tenure:', min_value=1, max_value=30, step=1)

        # Section 2: Products And Services
        with col2:
            st.header('Products And Services')
            phone_service = st.selectbox('PhoneService:', options=['Yes', 'No'])
            multiple_lines = st.selectbox('MultipleLines:', options=['Yes', 'No', 'No phone service'])
            internet_service = st.selectbox('InternetService:', options=['DSL', 'Fiber Optic', 'No internet service'])
            online_security = st.selectbox('OnlineSecurity:', options=['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox('OnlineBackup:', options=['Yes', 'No', 'No internet service'])

        # Section 3: Contract, Payment And Cost
        st.header('Contract, Payment And Cost')
        col3, col4 = st.columns(2)
        with col3:
            device_protection = st.selectbox('DeviceProtection:', options=['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox('TechSupport:', options=['Yes', 'No', 'No internet service'])
            streaming_tv = st.selectbox('StreamingTv:', options=['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox('StreamingMovies:', options=['Yes', 'No', 'No internet service'])

        with col4:
            contract = st.selectbox('Contract:', options=['Month-to-Month', 'One year', 'Two years'])
            paperless_billing = st.selectbox('PaperlessBilling:', options=['Yes', 'No', 'No internet service'])
            payment_method = st.selectbox('PaymentMethod:', options=['Mailed Check', 'Electronic Check', 'Bank Transfer', 'Credit Card'])
            monthly_charges = st.number_input('MonthlyCharges:')
            total_charges = st.number_input('TotalCharges:')

            make_predictions = None
            user_inputs = None

            # Prediction code
            if st.form_submit_button('Predict'):
                user_inputs = pd.DataFrame([[customer_ID, gender, senior_citizen, partner, dependents, tenure, phone_service,
                                             multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv,
                                             streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]],
                                            columns=['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService',
                                                     'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

                # Debugging: Display user inputs
                st.write("User Inputs:")
                st.write(user_inputs)

                # Preprocess user inputs if needed
                try:
                    # Assume preprocessor is part of the model pipeline
                    if model_choice == 'Gradient Boosting':
                        prediction_probabilities = gradient_model.predict_proba(user_inputs)
                    else:  # Random Forest
                        prediction_probabilities = random_forest.predict_proba(user_inputs)

                    churn_probability = prediction_probabilities[0][1]
                    make_predictions = gradient_model.predict(user_inputs) if model_choice == 'Gradient Boosting' else random_forest.predict(user_inputs)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                if make_predictions is not None:
                    prediction_result = "Churned" if make_predictions[0] == 1 else "Not Churned"
                    st.success(f"The predicted churn status is: {prediction_result}")
                    st.info(f"The probability of the customer not churning is: {1 - churn_probability:.2%}")
                else:
                    st.error("Prediction could not be made.")

                if user_inputs is not None:
                    save_to_history(user_inputs, make_predictions, model_choice)

if __name__ == "__main__":
    customer_details()
