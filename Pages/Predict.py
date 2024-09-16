import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title='Predict',
    page_icon='',
    layout='wide'
)

# Load models with error handling
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

gradient_model = load_model(r"models\gradient_descent_model")
random_forest = load_model(r"models\random_forest")

def save_to_history(inputs, predicted_outcome, selected_model):
    # Load or create history dataframe in session state
    if 'history_df' not in st.session_state:
        st.session_state.history_df = pd.DataFrame(columns=['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService',
                                                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'PredictedOutcome', 'SelectedModel'])

    # Append inputs and predicted outcome to history dataframe
    inputs['PredictedOutcome'] = predicted_outcome
    inputs['SelectedModel'] = selected_model
    st.session_state.history_df = pd.concat([st.session_state.history_df, inputs], ignore_index=True)

def validate_inputs(inputs):
    # Add checks to ensure inputs are within expected ranges
    if not (1 <= inputs['Tenure'] <= 30):
        st.error('Tenure must be between 1 and 30.')
        return False
    if inputs['MonthlyCharges'] <= 0 or inputs['TotalCharges'] < 0:
        st.error('Charges must be positive values.')
        return False
    return True

def customer_details():
    st.title('Customer Churn Prediction Page')
    st.write('Make your predictions here 👇')

    model_choice_column, _ = st.columns([1, 3])
    with model_choice_column:
        model_choice = st.selectbox('Select Model', ('Gradient Boosting', 'Random Forest'))

    with st.form('Customer_information'):
        st.header('Personal Details')
        col1, col2 = st.columns(2)
        with col1:
            customer_ID = st.number_input('CustomerID:', min_value=0)
            gender = st.selectbox('Gender:', options=['Male', 'Female'])
            senior_citizen = st.selectbox('SeniorCitizen:', options=['Yes', 'No'])
            partner = st.selectbox('Partner:', options=['Yes', 'No'])
            dependents = st.selectbox('Dependents:', options=['Yes', 'No'])
            tenure = st.number_input('Tenure:', min_value=1, max_value=30, step=1)

        with col2:
            st.header('Products And Services')
            phone_service = st.selectbox('PhoneService:', options=['Yes', 'No'])
            multiple_lines = st.selectbox('MultipleLines:', options=['Yes', 'No', 'No phone service'])
            internet_service = st.selectbox('InternetService:', options=['DSL', 'Fiber Optic', 'No internet service'])
            online_security = st.selectbox('OnlineSecurity:', options=['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox('OnlineBackup:', options=['Yes', 'No', 'No internet service'])

        st.header('Contract, Payment And Cost')
        col3, col4 = st.columns(2)
        with col3:
            device_protection = st.selectbox('DeviceProtection:', options=['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox('TechSupport:', options=['Yes', 'No', 'No internet service'])
            streaming_tv = st.selectbox('StreamingTV:', options=['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox('StreamingMovies:', options=['Yes', 'No', 'No internet service'])

        with col4:
            contract = st.selectbox('Contract:', options=['Month-to-Month', 'One year', 'Two years'])
            paperless_billing = st.selectbox('PaperlessBilling:', options=['Yes', 'No'])
            payment_method = st.selectbox('PaymentMethod:', options=['Mailed Check', 'Electronic Check', 'Bank Transfer', 'Credit Card'])
            monthly_charges = st.number_input('MonthlyCharges:', min_value=0.0)
            total_charges = st.number_input('TotalCharges:', min_value=0.0)

        make_predictions = None
        user_inputs = None

        if st.form_submit_button('Predict'):
            user_inputs = pd.DataFrame([[customer_ID, gender, senior_citizen, partner, dependents, tenure, phone_service,
                                         multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv,
                                         streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]], 
                                        columns=['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService',
                                                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

            st.write("User Inputs:")
            st.write(user_inputs)

            if validate_inputs({
                'Tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }):
                try:
                    if model_choice == 'Gradient Boosting':
                        model = gradient_model
                    else:
                        model = random_forest

                    if model is not None:
                        prediction_probabilities = model.predict_proba(user_inputs)
                        churn_probability = prediction_probabilities[0][1]
                        make_predictions = model.predict(user_inputs)
                        prediction_result = "Churned" if make_predictions[0] == 1 else "Not Churned"
                        st.success(f"The predicted churn status is: {prediction_result}")
                        st.info(f"The probability of the customer not churning is: {1 - churn_probability:.2%}")
                    else:
                        st.error("Selected model is not available.")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

                if user_inputs is not None and make_predictions is not None:
                    save_to_history(user_inputs, make_predictions, model_choice)

if __name__ == "__main__":
    customer_details()
