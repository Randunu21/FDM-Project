import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load the trained model
with open('fdm_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define encoders (should match what was used during training)
ordinal_encoder_credit_mix = OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])
le_score = LabelEncoder()

# App title and description
st.title("Credit Score Predictor")
st.write("Enter the following details to predict your credit score:")

# Input fields for necessary features
annual_income = st.number_input('Annual Income (in USD)', min_value=0.0, format="%.2f")
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, step=1)
num_credit_card = st.number_input('Number of Credit Cards', min_value=0, step=1)
interest_rate = st.slider('Interest Rate (%)', 0.0, 50.0, 5.0)
num_of_loan = st.number_input('Number of Loans', min_value=0, step=1)
delay_from_due_date = st.slider('Delay from Due Date (days)', 0, 60, 0)
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)

# Categorical fields
credit_mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
occupation = st.selectbox('Occupation', ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
                                         'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
                                         'Scientist', 'Teacher', 'Writer'])

outstanding_debt = st.number_input('Outstanding Debt (in USD)', min_value=0.0, format="%.2f")
credit_history_age = st.number_input('Credit History Age (years)', min_value=0, step=1)
total_emi_per_month = st.number_input('Total EMI per Month (in USD)', min_value=0.0, format="%.2f")

# Feature Engineering
# Total number of accounts
total_num_accounts = num_bank_accounts + num_credit_card

# Debt-to-Income Ratio
debt_to_income_ratio = outstanding_debt / annual_income if annual_income > 0 else 0

# Encoding the inputs
credit_mix_encoded = ordinal_encoder_credit_mix.fit_transform(np.array([[credit_mix]]))[0][0]

# One-hot encoding the occupation
occupation_list = ['Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
                   'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
                   'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
                   'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer']

occupation_encoded = [1 if f'Occupation_{occupation}' == occ else 0 for occ in occupation_list]

# Prepare the input data as a numpy array
input_data = np.array([[annual_income, num_bank_accounts, num_credit_card, interest_rate, num_of_loan, 
                        delay_from_due_date, num_of_delayed_payment, num_credit_inquiries, credit_mix_encoded, 
                        outstanding_debt, credit_history_age, total_emi_per_month, total_num_accounts, 
                        debt_to_income_ratio]])

# Add the one-hot encoded occupation columns
input_data = np.concatenate([input_data, np.array([occupation_encoded])], axis=1)

# Ensure the input data is 2D
input_data = input_data.reshape(1, -1)

# Mapping for Credit Score Output
credit_score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}

# When the user clicks 'Predict'
if st.button('Predict Credit Score'):
    prediction = model.predict(input_data)  # Use your prediction logic
    mapped_prediction = credit_score_mapping.get(prediction[0], "Unknown")
    st.success(f'Predicted Credit Score: {mapped_prediction}')