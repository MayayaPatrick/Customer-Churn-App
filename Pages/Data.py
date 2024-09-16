import streamlit as st
import pandas as pd

# Set up the page configuration
st.set_page_config(
    page_title='View Data',
    page_icon='',
    layout='wide'
)

def main():
    st.title('DATASETS')
    st.write('This page contains data from an SQL database that was used in this project')

    # Load data from CSV
    data_file = "data/Telco-churn-first-3000.txt"  # Use forward slashes for paths
    try:
        telco_churn_first_3000 = pd.read_csv(data_file)
    except FileNotFoundError:
        st.error(f"File not found: {data_file}")
        telco_churn_first_3000 = pd.DataFrame()  # Empty DataFrame as a fallback
    except pd.errors.EmptyDataError:
        st.error(f"The file {data_file} is empty.")
        telco_churn_first_3000 = pd.DataFrame()
    except pd.errors.ParserError:
        st.error(f"Error parsing the file {data_file}.")
        telco_churn_first_3000 = pd.DataFrame()

    def display_section(section):
        if telco_churn_first_3000.empty:
            st.write("No data to display.")
            return
        
        if section == "Categorical":
            st.write("Categorical Section:")
            st.dataframe(telco_churn_first_3000.select_dtypes(include=['object']))  # Interactive DataFrame display
        elif section == "Numerical":
            st.write("Numerical Section:")
            st.dataframe(telco_churn_first_3000.select_dtypes(exclude=['object']))  # Interactive DataFrame display

    dataset_column, section_column = st.columns([1, 1])

    with dataset_column:
        selected_dataset_name = st.selectbox("Select Dataset", ["All Datasets"])  # Add more options if needed

    with section_column:
        selected_section = st.selectbox("Select Section", ["Categorical", "Numerical"])

    # Display the selected dataset and section
    st.write("Selected Dataset:", selected_dataset_name)
    if selected_section:
        display_section(selected_section)
    else:
        st.write("Please select a section to display.")

if __name__ == "__main__":
    main()
