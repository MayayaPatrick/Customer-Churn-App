import streamlit as st
import plotly.express as px
import pandas as pd

# Function to load data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("https://raw.githubusercontent.com/MayayaPatrick/Churn-rate-Prediction-Model/main/notebooks/train.csv")
        
        # Replace 1 with 'Yes' and 0 with 'No' in the Churn column
        data['Churn'] = data['Churn'].replace({1: 'Yes', 0: 'No'})
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Function for EDA Dashboard with added filters and varied colors
def eda_dashboard(df):
    st.markdown('#### Univariate Analysis')

    # Sidebar Filters
    st.sidebar.markdown("### Filters")
    churn_filter = st.sidebar.multiselect('Select Churn Status', df['Churn'].unique(), default=df['Churn'].unique())
    contract_filter = st.sidebar.multiselect('Select Contract Type', df['Contract'].unique(), default=df['Contract'].unique())
    gender_filter = st.sidebar.multiselect('Select Gender', df['gender'].unique(), default=df['gender'].unique())
    internet_service_filter = st.sidebar.multiselect('Select Internet Service', df['InternetService'].unique(), default=df['InternetService'].unique())
    payment_method_filter = st.sidebar.multiselect('Select Payment Method', df['PaymentMethod'].unique(), default=df['PaymentMethod'].unique())
    senior_filter = st.sidebar.multiselect('Select Senior Citizen Status', df['SeniorCitizen'].unique(), default=df['SeniorCitizen'].unique())

    # Filter data based on selections
    filtered_df = df[(df['Churn'].isin(churn_filter)) &
                     (df['Contract'].isin(contract_filter)) &
                     (df['gender'].isin(gender_filter)) &
                     (df['InternetService'].isin(internet_service_filter)) &
                     (df['PaymentMethod'].isin(payment_method_filter)) &
                     (df['SeniorCitizen'].isin(senior_filter))]

    # Add univariate analysis charts
    col1, col2, col3 = st.columns(3)
    
    # Monthly Charges Histogram
    with col1:
        st.markdown("**Monthly Charges**")
        fig = px.histogram(filtered_df, x="MonthlyCharges", nbins=20, title="Monthly Charges Distribution",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

    # Tenure Histogram
    with col2:
        st.markdown("**Tenure**")
        fig = px.histogram(filtered_df, x="tenure", nbins=20, title="Tenure Distribution",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

    # Total Charges Histogram
    with col3:
        st.markdown("**Total Charges**")
        fig = px.histogram(filtered_df, x="TotalCharges", nbins=20, title="Total Charges Distribution",
                           color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

    st.divider()

    st.markdown('#### Bivariate Analysis')

    # Pie chart with Blue Sky and Mango colors for Churn
    col1, col2 = st.columns(2)
    with col1:
        pieplot = px.pie(filtered_df, names="Churn", title=f"Churn Rate for {len(filtered_df)} customers", 
                         color="Churn", color_discrete_sequence=['#87CEEB', '#FFB347'])  # BlueSky and Mango
        st.plotly_chart(pieplot)

    # Bar chart with Green and Orange for Churn
    with col2:
        churn_internet_counts = filtered_df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        fig = px.bar(churn_internet_counts, x='InternetService', y='count', color='Churn', barmode='group',
                     color_discrete_sequence=['#32CD32', '#FFA500'])  # Green and Orange
        fig.update_layout(
            xaxis_title='Internet Service',
            yaxis_title='Count',
            title=f'Churn by Internet Service for selected filters'
        )
        st.plotly_chart(fig)

    st.divider()

    # Multivariate Analysis
    st.markdown('#### Multivariate Analysis')

    # Scatterplot for tenure and monthly charges over churn with distinct colors
    fig = px.scatter(filtered_df, x='tenure', y='MonthlyCharges',
                     title=f'Tenure vs Monthly Charges Distribution ({len(filtered_df)} customers)',
                     color='Churn', color_discrete_sequence=px.colors.diverging.Picnic)
    fig.update_layout(width=600, height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig)

    st.divider()

    # Contract Types by Churn with Green and Orange
    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(filtered_df, "Contract", color="Churn", title="Contract Types by Churn",
                           color_discrete_sequence=['#32CD32', '#FFA500'])  # Green and Orange
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig)

    # Payment Methods by Churn with Green and Orange
    with col4:
        fig = px.histogram(filtered_df, "PaymentMethod", color="Churn", title="Payment Methods by Churn",
                           color_discrete_sequence=['#32CD32', '#FFA500'])  # Green and Orange
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig)

# Main function
def main():
    df = load_data()
    if df.empty:
        st.stop()  # Stop the app if data cannot be loaded

    st.title('Dashboard')

    # Add interactive selection for the dashboard type
    col1, col2 = st.columns(2)
    with col1:
        dashboard_type = st.selectbox('Select Type of Dashboard', options=['EDA', 'KPI'])

    if dashboard_type == 'EDA':
        eda_dashboard(df)
    else:
        # Add your KPI dashboard function if needed
        pass

if __name__ == "__main__":
    main()
