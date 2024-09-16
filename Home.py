import streamlit as st

# Function to display the home page
def show_home_page():
    st.title("Customer Churn Prediction App")
    st.markdown("""
    This app uses a machine learning model to predict if a customer will churn based on their order quantity and information.
    """)

    st.subheader("Key Features")
    st.markdown("""
    - Upload your CSV file containing customer data
    - Select your desired features for classification
    - Choose a machine learning model from the dropdown menu
    - Click 'Classify' to get the predicted result
    - The app also provides a detailed report on the performance of the model
    """)

    st.subheader("App Features")
    st.markdown("""
    - **Data-driven Decisions**: Make informed decisions backed by data analytics
    - **Easy Machine Learning**: Utilize powerful machine learning algorithms effortlessly
    - **Live Demo**: Watch a demo video to see the app in action
    """)

    st.markdown("[Watch a demo video](#)")

    st.subheader("How to Run the Application")
    st.code("""
    # Activate virtual environment
    # env/Scripts/activate
    streamlit run Home.py
    """, language="bash")

    st.subheader("Machine Learning Integration")
    st.markdown("""
    **Model Selection**: Choose advanced models for accurate predictions.
    **Seamless Integration**: Integrate predictions into your workflow with a user-friendly interface.
    """)

    st.subheader("Need Help?")
    st.markdown("For collaboration contact me at [pmayaya55@gmail.com](mailto:pmayaya55@gmail.com).")
    st.button("Repository on GitHub", help="Visit the GitHub repository")

# Function to render the selected page
def render_page(page_name):
    if page_name == "Home":
        show_home_page()
    elif page_name == "Data":
        import PSages.Data as Data
        Data.main()
    elif page_name == "Dashboard":
        import Pages.Dashboard as Dashboard
        Dashboard.main()
    elif page_name == "Predict":
        import Pages.Predict as Predict
        Predict.main()
    elif page_name == "History":
        import Pages.History as History
        History.main()

# Main function
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Customer Churn App",
        page_icon=":phone:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

   
    # Navigation menu
    st.sidebar.title("Navigation")
    st.session_state["page"] = st.sidebar.selectbox(
        "Select page", 
        ["Home", "Data", "Dashboard", "Predict", "History"]
    )

    # Render the selected page
    render_page(st.session_state["page"])

# Run the main function
if __name__ == "__main__":
    main()
