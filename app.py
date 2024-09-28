import streamlit as st
import numpy as np
import pickle  # or pickle if you used pickle to save the model


# way to load the pickle file
with open('fdm_model.pickle', 'rb') as file:  # 'rb' means read in binary mode
    model = pickle.load(file)  # Load the model correctly


# App title and description
st.title("Credit Score Predictor")
st.write("Enter the following details to predict your credit score:")

# Input fields
age = st.number_input('Age', min_value=18, max_value=100, step=1)
annual_income = st.number_input('Annual Income (in USD)', min_value=0.0, format="%.2f")
monthly_inhand_salary = st.number_input('Monthly Inhand Salary (in USD)', min_value=0.0, format="%.2f")
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, step=1)
num_credit_card = st.number_input('Number of Credit Cards', min_value=0, step=1)
interest_rate = st.slider('Interest Rate (%)', 0.0, 50.0, 5.0)
num_of_loan = st.number_input('Number of Loans', min_value=0, step=1)
delay_from_due_date = st.slider('Delay from Due Date (days)', 0, 60, 0)
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
changed_credit_limit = st.number_input('Changed Credit Limit (%)', min_value=-100.0, format="%.2f")
num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)
credit_mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
outstanding_debt = st.number_input('Outstanding Debt (in USD)', min_value=0.0, format="%.2f")
credit_utilization_ratio = st.slider('Credit Utilization Ratio (%)', 0.0, 100.0, 30.0)
credit_history_age = st.number_input('Credit History Age (years)', min_value=0, step=1)
payment_of_min_amount = st.selectbox('Payment of Minimum Amount', ['Yes', 'No'])
total_emi_per_month = st.number_input('Total EMI per Month (in USD)', min_value=0.0, format="%.2f")
amount_invested_monthly = st.number_input('Amount Invested Monthly (in USD)', min_value=0.0, format="%.2f")
payment_behaviour = st.selectbox('Payment Behaviour', ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments'])
monthly_balance = st.number_input('Monthly Balance (in USD)', min_value=0.0, format="%.2f")
total_num_accounts = st.number_input('Total Number of Accounts', min_value=0, step=1)
debt_per_account = st.number_input('Debt Per Account (in USD)', min_value=0.0, format="%.2f")
debt_to_income_ratio = st.number_input('Debt to Income Ratio (%)', min_value=0.0, format="%.2f")
delayed_payments_per_account = st.number_input('Delayed Payments Per Account', min_value=0.0, format="%.2f")
total_monthly_expenses = st.number_input('Total Monthly Expenses (in USD)', min_value=0.0, format="%.2f")

# Prepare the input data as a numpy array
input_data = np.array([[age, annual_income, monthly_inhand_salary, num_bank_accounts, num_credit_card,
                        interest_rate, num_of_loan, delay_from_due_date, num_of_delayed_payment,
                        changed_credit_limit, num_credit_inquiries, credit_mix, outstanding_debt,
                        credit_utilization_ratio, credit_history_age, payment_of_min_amount,
                        total_emi_per_month, amount_invested_monthly, payment_behaviour,
                        monthly_balance, total_num_accounts, debt_per_account, debt_to_income_ratio,
                        delayed_payments_per_account, total_monthly_expenses]])

# When the user clicks 'Predict'
if st.button('Predict Credit Score'):
    prediction = model.predict(input_data)  # Replace with your prediction logic
    st.success(f'Predicted Credit Score: {prediction[0]}')

# Additional styling for attractiveness
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    color: #333;
    font-family: Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)