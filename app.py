import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load the trained model
with open('fdm_model.pickle', 'rb') as file:
    model = pickle.load(file)


# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define encoders (should match what was used during training)
ordinal_encoder_credit_mix = OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])
le_score = LabelEncoder()

# App title and description
st.title("Credit Score Predictor")
st.write("Enter the following details to predict your credit score:")

# Input fields for necessary features
Annual_Income = st.number_input('Annual Income (in USD)', min_value=0.0, format="%.2f")
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, step=1)
num_credit_card = st.number_input('Number of Credit Cards', min_value=0, step=1)
Interest_Rate = st.slider('Interest Rate (%)', 0.0, 50.0, 5.0)
Num_of_Loan = st.number_input('Number of Loans', min_value=0, step=1)
delay_from_due_date = st.slider('Delay from Due Date (days)', 0, 60, 0)
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)

# Categorical fields
credit_mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
occupation = st.selectbox('Occupation', ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
                                         'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
                                         'Scientist', 'Teacher', 'Writer'])


# Payment of Minimum Amount field (yes = 2, no = 1, NM = 0)
payment_of_min_amount = st.selectbox('Payment of Minimum Amount', ['yes', 'no', 'NM'])
payment_of_min_amount = {'yes': 2, 'no': 1, 'NM': 0}[payment_of_min_amount]


outstanding_debt = st.number_input('Outstanding Debt (in USD)', min_value=0.0, format="%.2f")
credit_history_age = st.number_input('Credit History Age (years)', min_value=0, step=1)

# Feature Engineering
# Total number of accounts
total_num_accounts = num_bank_accounts + num_credit_card

# Debt-to-Income Ratio
# debt_to_income_ratio = outstanding_debt / annual_income if annual_income > 0 else 0

# Encoding the inputs
credit_mix_encoded = ordinal_encoder_credit_mix.fit_transform(np.array([[credit_mix]]))[0][0]

# One-hot encoding the occupation
occupation_list = ['Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
                   'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
                   'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
                   'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer']

occupation_encoded = [1 if f'Occupation_{occupation}' == occ else 0 for occ in occupation_list]

# Prepare the numerical inputs (without the one-hot encoded occupation)
numerical_features = np.array([[Annual_Income, Interest_Rate, Num_of_Loan, 
                                delay_from_due_date, num_of_delayed_payment, num_credit_inquiries,
                                outstanding_debt, credit_history_age, total_num_accounts]])

# Convert the numerical features to a DataFrame before scaling
numerical_features_scaled = pd.DataFrame(numerical_features, columns=[
    'Annual_Income', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
    'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
    'Credit_History_Age', 'Total_Num_Accounts'
])

numerical_features_scaled = scaler.transform(numerical_features_scaled)

# numerical_features_scaled = numerical_features_scaled.to_numpy()



# Prepare the final input in the specified order
final_input = np.concatenate([
    numerical_features_scaled[:, :6],  # First 6 numerical features (0 to 5)
    np.array([[credit_mix_encoded]]),   # Credit mix as a 2D array with shape (1, 1)
    numerical_features_scaled[:, 6:8],  # Next numerical features (6 to 8)
    np.array([[payment_of_min_amount]]), # Payment of min amount as a 2D array with shape (1, 1)
    np.array(occupation_encoded).reshape(1, -1),  # Ensure occupation is a 2D array with shape (1, n)
    numerical_features_scaled[:, -1:]     # Total number of accounts as a 2D array with shape (1, 1)
], axis=1)  # Concatenate along columns (axis=1)

# Ensure the input data is 2D
final_input = final_input.reshape(1, -1)


# Mapping for Credit Score Output
credit_score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}

# When the user clicks 'Predict'
if st.button('Predict Credit Score'):
    prediction = model.predict(final_input)  # Use your prediction logic
    mapped_prediction = credit_score_mapping.get(prediction[0], "Unknown")
    st.success(f'Predicted Credit Score: {mapped_prediction}')
    st.write(final_input)


