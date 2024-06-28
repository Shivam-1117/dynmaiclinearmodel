import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score

# Page title
st.set_page_config(layout='wide', page_title='Dynamic Linear Modelling App', page_icon='🏗️')
st.title('🏗️ Dynamic Linear Model')
sleep_time = 1

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')


# Sidebar for accepting input parameters
st.header('1.1. Input Raw Data')
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file is not None:
    retail_data = pd.read_excel(uploaded_file, index_col = False)


# Initiate the model building process
if uploaded_file: 
    with st.status("Running ...", expanded=True) as status:
        st.write("Preparing data ...")
        time.sleep(sleep_time)
    X = retail_data[['Digital (Million $)', 'radio (Million $)', 'TV (Thousands $)']]
    y = retail_data['sales (Million $)']
    st.write("Done.")

    st.header('2. Model Configuration')
    to_test_options = ['Not in Model', 'In Model', 'Outside Model']
    var_type = ['Linear', 'Adstock']
    lag_options = [0, 1, 2, 3, 4, 5]

    # Create a dictionary to hold the user inputs
    user_inputs = {}

    # Create the header row
    header_cols = st.columns(9)
    headers = ['Variable', 'In Model', 'Variable Type', 'Lag Min', 'Lag Max', 'Decay Steps', 'Decay Min', 'Decay Max', 'Discount Factor']
    for col, header in zip(header_cols, headers):
        col.write(f"**{header}**")

    # Create the input rows
    with st.form(key='my_form'):
        for var in X.columns:
            input_cols = st.columns(9)
            input_cols[0].write(f"**{var}**")
            in_model = input_cols[1].selectbox('', to_test_options, key=f'{var}_in_model')
            vtype = input_cols[2].selectbox('', var_type, key=f'{var}_vtype')
            lag_min = input_cols[3].selectbox('', lag_options, key=f'{var}_lag_min')
            lag_max = input_cols[4].selectbox('', lag_options, key=f'{var}_lag_max')
            decay_steps = input_cols[5].number_input('', value=1, key=f'{var}_decay_steps', step=1)
            decay_min = input_cols[6].number_input('', value=1.0, key=f'{var}_decay_min', step=0.01, format="%.2f")
            decay_max = input_cols[7].number_input('', value=1.0, key=f'{var}_decay_max', step=0.01, format="%.2f")
            discount_factor = input_cols[8].number_input('', value=1.0, key=f'{var}_discount_factor', step=0.0001, format="%.4f")
            
            user_inputs[var] = {
                'Variable': var,
                'In Model': in_model,
                'Variable Type': vtype,
                'Lag Min': lag_min,
                'Lag Max': lag_max,
                'Decay Steps': decay_steps,
                'Decay Min': decay_min,
                'Decay Max': decay_max,
                'Discount Factor': discount_factor
            }

        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # When the form is submitted
    if submit_button:
        # Convert the user inputs dictionary to a DataFrame
        with st.status("Running ...", expanded=True) as status:
            st.write("Setting up the model configuration ...")
            time.sleep(sleep_time)
        model_params = pd.DataFrame.from_dict(user_inputs, orient='index').reset_index(drop = True)
        st.write('### Collected Model Parameters')
        st.dataframe(model_params)
    

        
    st.write("Evaluating performance metrics ...")
    time.sleep(sleep_time)
    
    
    st.write("Displaying performance metrics ...")
    time.sleep(sleep_time)

    
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
