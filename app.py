import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Set page config for Streamlit app
st.set_page_config(page_title='Autism Detection App', page_icon='🤖', layout='wide')

# Title of the application
st.title("🧠 Autism Detection Using Machine Learning")

# Load pre-trained models (saved using joblib)
dec_model = joblib.load('./models/decision_tree_model.joblib')
rndm_model = joblib.load('./models/random_forest_model.joblib')
svm_model = joblib.load('./models/svm_model.joblib')
knn_model = joblib.load('./models/knn_model.joblib')
nb_model = joblib.load('./models/naive_bayes_model.joblib')
lr_model = joblib.load('./models/logistic_regression_model.joblib')
best_clf = joblib.load('./models/optimized_svm_model.joblib')

# Load the fitted MinMaxScaler from the training phase
scaler = joblib.load('./models/scaler.joblib')

# Sidebar for model selection
st.sidebar.header('Select a Model')
model_choice = st.sidebar.selectbox('Choose a model', ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'Logistic Regression', 'Optimized SVM'])

# Sidebar for instructions
st.sidebar.markdown('''
### Instructions:
1. Enter the patient details on the right side.
2. Select the desired model from the sidebar.
3. Click **Predict** to see if ASD is detected.
''')

# Define expected features after one-hot encoding
expected_columns = ['age', 'result', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 
                    'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                    'gender_f', 'gender_m', 'ethnicity_?', 'ethnicity_Asian', 'ethnicity_Black', 
                    'ethnicity_Hispanic', 'ethnicity_Latino', 'ethnicity_Middle Eastern ', 
                    'ethnicity_Others', 'ethnicity_Pasifika', 'ethnicity_South Asian', 
                    'ethnicity_Turkish', 'ethnicity_White-European', 'ethnicity_others', 
                    'jundice_no', 'jundice_yes', 'austim_no', 'austim_yes',
                    'contry_of_res_Afghanistan', 'contry_of_res_AmericanSamoa', 'contry_of_res_Angola', 
                    'contry_of_res_Argentina', 'contry_of_res_Armenia', 'contry_of_res_Aruba', 
                    'contry_of_res_Australia', 'contry_of_res_Austria', 'contry_of_res_Azerbaijan', 
                    'contry_of_res_Bahamas', 'contry_of_res_Bangladesh', 'contry_of_res_Belgium', 
                    'contry_of_res_Bolivia', 'contry_of_res_Brazil', 'contry_of_res_Burundi', 
                    'contry_of_res_Canada', 'contry_of_res_Chile', 'contry_of_res_China', 
                    'contry_of_res_Costa Rica', 'contry_of_res_Cyprus', 'contry_of_res_Czech Republic', 
                    'contry_of_res_Ecuador', 'contry_of_res_Egypt', 'contry_of_res_Ethiopia', 
                    'contry_of_res_Finland', 'contry_of_res_France', 'contry_of_res_Germany', 
                    'contry_of_res_Hong Kong', 'contry_of_res_Iceland', 'contry_of_res_India', 
                    'contry_of_res_Indonesia', 'contry_of_res_Iran', 'contry_of_res_Iraq', 
                    'contry_of_res_Ireland', 'contry_of_res_Italy', 'contry_of_res_Japan', 
                    'contry_of_res_Jordan', 'contry_of_res_Kazakhstan', 'contry_of_res_Lebanon', 
                    'contry_of_res_Malaysia', 'contry_of_res_Mexico', 'contry_of_res_Nepal', 
                    'contry_of_res_Netherlands', 'contry_of_res_New Zealand', 'contry_of_res_Nicaragua', 
                    'contry_of_res_Niger', 'contry_of_res_Oman', 'contry_of_res_Pakistan', 
                    'contry_of_res_Philippines', 'contry_of_res_Portugal', 'contry_of_res_Romania', 
                    'contry_of_res_Russia', 'contry_of_res_Saudi Arabia', 'contry_of_res_Serbia', 
                    'contry_of_res_Sierra Leone', 'contry_of_res_South Africa', 'contry_of_res_Spain', 
                    'contry_of_res_Sri Lanka', 'contry_of_res_Sweden', 'contry_of_res_Tonga', 
                    'contry_of_res_Turkey', 'contry_of_res_Ukraine', 'contry_of_res_United Arab Emirates', 
                    'contry_of_res_United Kingdom', 'contry_of_res_United States', 'contry_of_res_Uruguay', 
                    'contry_of_res_Viet Nam', 'relation_?', 'relation_Health care professional', 
                    'relation_Others', 'relation_Parent', 'relation_Relative', 'relation_Self']


# Function to preprocess user input
def preprocess_input(user_input):
    # One-hot encode categorical features
    encoded_data = pd.get_dummies(user_input, columns=['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'relation'])
    
    # Add missing columns with zero values
    for col in expected_columns:
        if col not in encoded_data.columns:
            encoded_data[col] = 0

    # Ensure columns are in the same order as expected
    encoded_data = encoded_data[expected_columns]

    # Use the loaded scaler to transform 'age' and 'result'
    encoded_data[['age', 'result']] = scaler.transform(encoded_data[['age', 'result']])

    return encoded_data

# Layout for user input
st.markdown("### Patient Details")
col1, col2 = st.columns(2)

# Collect user inputs in a well-organized layout
with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    gender = st.selectbox('Gender', ['f', 'm'])
    ethnicity = st.selectbox('Ethnicity', ['White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'others'])
    jundice = st.selectbox('Jaundice History', ['no', 'yes'])
    austim = st.selectbox('Family Autism History', ['no', 'yes'])
    contry_of_res = st.selectbox('Country of Residence', ['United States', 'Brazil', 'Spain', 'Egypt', 'New Zealand', 'Others'])
    relation = st.selectbox('Relation', ['Self', 'Parent', 'Health care professional', 'Relative', 'Others'])

with col2:
    result = st.number_input('Result (Score)', min_value=0, max_value=10, value=5)

    # Collect responses for A1_Score to A10_Score in a more compact dropdown
    A1_Score = st.selectbox('A1_Score', [1, 0])
    A2_Score = st.selectbox('A2_Score', [1, 0])
    A3_Score = st.selectbox('A3_Score', [1, 0])
    A4_Score = st.selectbox('A4_Score', [1, 0])
    A5_Score = st.selectbox('A5_Score', [1, 0])
    A6_Score = st.selectbox('A6_Score', [1, 0])
    A7_Score = st.selectbox('A7_Score', [1, 0])
    A8_Score = st.selectbox('A8_Score', [1, 0])
    A9_Score = st.selectbox('A9_Score', [1, 0])
    A10_Score = st.selectbox('A10_Score', [1, 0])

# Collect all user input into a DataFrame
user_input = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'ethnicity': [ethnicity],
    'jundice': [jundice],
    'austim': [austim],
    'contry_of_res': [contry_of_res],
    'result': [result],
    'relation': [relation],
    'A1_Score': [A1_Score],
    'A2_Score': [A2_Score],
    'A3_Score': [A3_Score],
    'A4_Score': [A4_Score],
    'A5_Score': [A5_Score],
    'A6_Score': [A6_Score],
    'A7_Score': [A7_Score],
    'A8_Score': [A8_Score],
    'A9_Score': [A9_Score],
    'A10_Score': [A10_Score]
})

# Preprocess the user input
preprocessed_input = preprocess_input(user_input)

# Predict and display the result when the button is clicked
if st.button('Predict'):
    if model_choice == 'Decision Tree':
        prediction = dec_model.predict(preprocessed_input)
    elif model_choice == 'Random Forest':
        prediction = rndm_model.predict(preprocessed_input)
    elif model_choice == 'SVM':
        prediction = svm_model.predict(preprocessed_input)
    elif model_choice == 'KNN':
        prediction = knn_model.predict(preprocessed_input)
    elif model_choice == 'Naive Bayes':
        prediction = nb_model.predict(preprocessed_input)
    elif model_choice == 'Logistic Regression':
        prediction = lr_model.predict(preprocessed_input)
    elif model_choice == 'Optimized SVM':
        prediction = best_clf.predict(preprocessed_input)
    
    # Display the result in a visually appealing way
    if prediction == 1:
        st.success('Prediction: ASD Detected ✅')
    else:
        st.info('Prediction: No ASD Detected ❌')
