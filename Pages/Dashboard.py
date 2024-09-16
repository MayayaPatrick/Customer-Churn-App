import streamlit as st
import plotly.express as px
import pandas as pd

# Function to load data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("https://raw.githubusercontent.com/MayayaPatrick/Churn-rate-Prediction-Model/main/notebooks/train.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Function for EDA Dashboard with added filters
def eda_dashboard(df):
    st.markdown('#### Univariate Analysis')

    # Add filters for Churn and Contract Type
    churn_filter = st.sidebar.multiselect('Select Churn Status', df['Churn'].unique(), default=df['Churn'].unique())
    contract_filter = st.sidebar.multiselect('Select Contract Type', df['Contract'].unique(), default=df['Contract'].unique())
    filtered_df = df[(df['Churn'].isin(churn_filter)) & (df['Contract'].isin(contract_filter))]

    # Monthly charges histogram
    col1, col2, col3 = st.columns(3)
    with col1:
        monthlycharges_histogram = px.histogram(filtered_df, "MonthlyCharges", title="Distribution of Monthly Charges",
                                                color_discrete_sequence=['#EF553B'])
        monthlycharges_histogram.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(monthlycharges_histogram)

    # Tenure Histogram  
    with col2:
        tenure_histogram = px.histogram(filtered_df, "tenure", title="Distribution of Tenure",
                                        color_discrete_sequence=['#00CC96'])
        tenure_histogram.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(tenure_histogram)

    # Total charges histogram
    with col3:
        totalcharges_histogram = px.histogram(filtered_df, "TotalCharges", title="Distribution of Total Charges",
                                              color_discrete_sequence=['#636EFA'])
        totalcharges_histogram.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(totalcharges_histogram)

    st.divider()

    st.markdown('#### Bivariate Analysis')
    col4, col5 = st.columns(2)
    with col4:
        colors = ['blue', 'red']
        pieplot = px.pie(filtered_df, names="Churn", title=f"Churn Rate for {len(filtered_df)} customers", color="Churn",
                         color_discrete_sequence=colors)
        st.plotly_chart(pieplot)

    with col5:
        churn_gender_counts = filtered_df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        fig = px.bar(churn_gender_counts, x='InternetService', y='count', color='Churn', barmode='group',
                     color_discrete_map={'No': 'blue', 'Yes': 'red'})
        fig.update_layout(
            xaxis_title='Internet Service',
            yaxis_title='Count',
            title=f'Churn by Internet Service for selected filters'
        )
        st.plotly_chart(fig)

    st.divider()

    st.markdown('#### Multivariate Analysis')

    # Scatterplot for tenure and monthly charges over churn
    fig = px.scatter(filtered_df, x='tenure', y='MonthlyCharges',
                     title=f'Tenure vs Monthly Charges Distribution ({len(filtered_df)} customers)',
                     color='Churn', color_discrete_map={'Yes': 'Red', 'No': 'blue'})
    fig.update_layout(width=600, height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig)

    st.divider()

    col6, col7 = st.columns(2)
    with col6:
        fig = px.histogram(filtered_df, "Contract", color="Churn", title="Contract Types by Churn",
                           color_discrete_map={'Yes': 'red', 'No': 'blue'})
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig)

    with col7:
        fig = px.histogram(filtered_df, "PaymentMethod", color="Churn", title="Payment Methods by Churn",
                           color_discrete_map={'Yes': 'red', 'No': 'blue'})
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig)

# Define KPI dashboard function with dynamic filtering
def kpi_dashboard(df):
    st.markdown('## 📊 Key Performance Indicators')

    # Filters
    contract_filter = st.sidebar.multiselect('Select Contract Type', df['Contract'].unique(), default=df['Contract'].unique())
    filtered_df = df[df['Contract'].isin(contract_filter)]

    # Churn rate calculation
    churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    st.markdown(f'Churn Rate: **{churn_rate:.2f}%**')

    # Average monthly charges calculation
    average_monthly_charges = filtered_df['MonthlyCharges'].mean()
    st.markdown(f'Average Monthly Charges: **${average_monthly_charges:.2f}**')

    # Total customers calculation
    total_customers = len(filtered_df)
    st.markdown(f'Total Customers: **{total_customers}**')

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
        kpi_dashboard(df)

if __name__ == "__main__":
    main()
