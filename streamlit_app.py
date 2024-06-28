import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from pydlm import dlm, trend, dynamic


# Lag and Adstock Functions
def apply_lag(media_data, lag = 0):
  return media_data.shift(lag, fill_value = 0)

def apply_adstock(media_data, decay = 0):
  media_data_ad = []
  for i in range(len(media_data)):
    if i == 0:
      media_data_ad.append(media_data[i])
    else:
      media_data_ad.append(media_data[i] + (1 - decay) * media_data_ad[i-1])
  return media_data_ad

# Dynamic Commponents
def create_dynamic_comp(feature_data, feature_name, discount_factor = 1):
  features = [[feature_data[i]] for i in range(len(feature_data))]
  dynamic_comp = dynamic(features = features,
                         name = feature_name,
                         discount = discount_factor
                         )
  return dynamic_comp, features



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
    st.write("### Model Raw Data")
    st.dataframe(retail_data)

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
    with st.form(key='model_config'):
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
        if 'In Model' in model_params['In Model'].tolist() or 'Outside Model' in model_params['In Model'].tolist():
            
            st.write('### Collected Model Parameters')
            st.dataframe(model_params)

            # Creating the transformed dataset
            transformed_data = pd.DataFrame()
            outside_vars_dict = {}
            outside_vars = []
            not_outside_vars = []
            outside_vars_ids = {}

            for i in range(model_params.shape[0]):
                variable = model_params.loc[i, 'Variable']
                if model_params.loc[i, 'In Model'] == 'In Model':
                    not_outside_vars.append(variable)
                    if model_params.loc[i, 'Variable Type'] == 'Adstock':
                        transformed_data[variable] = apply_lag(X[variable], model_params.loc[i, 'Lag Min'])
                        transformed_data[variable] = apply_adstock(transformed_data[variable], model_params.loc[i, 'Decay Min'])
                    else:
                        transformed_data[variable] = X[variable]
                elif model_params.loc[i, 'In Model'] == 'Outside Model':
                    outside_vars.append(variable)
                    lags = range(model_params.loc[i, 'Lag Min'], model_params.loc[i, 'Lag Max'] + 1)
                    step = (model_params.loc[i, 'Decay Max'] - model_params.loc[i, 'Decay Min'])/model_params.loc[i, 'Decay Steps']
                    decays = np.arange(model_params.loc[i, 'Decay Min'], model_params.loc[i, 'Decay Max'] + step, step)
                    discount_factor = model_params.loc[i, 'Discount Factor']
                    id = 0
                    for lag in lags:
                        for decay in decays:
                            decay = round(decay, 2)
                            transformed_data[variable + '_' + str(lag) + '_' + str(decay)] = apply_lag(X[variable], lag)
                            transformed_data[variable + '_' + str(lag) + '_' + str(decay)] = apply_adstock(transformed_data[variable + '_' + str(lag) + '_' + str(decay)], decay)
                            if variable not in outside_vars_dict.keys():
                                outside_vars_dict[variable] = [[variable + '_' + str(lag) + '_' + str(decay), lag, decay, discount_factor]]
                            else:
                                outside_vars_dict[variable].append([variable + '_' + str(lag) + '_' + str(decay), lag, decay, discount_factor])
                        outside_vars_ids[variable] = list(range(len(outside_vars_dict[variable])))

            model_vars = not_outside_vars + list(outside_vars_dict.keys())
            
            # Creating Dynamic Components
            dynamic_comps = {}
            feature_dict = {}
            for variable in model_vars:
                if variable in list(outside_vars_dict.keys()):
                    for item in outside_vars_dict[variable]:
                        discount_factor = item[2]
                        dynamic_comps[item[0]], feature_dict[item[0]] = create_dynamic_comp(transformed_data[item[0]], item[0], discount_factor)
                else:
                    discount_factor = model_params[model_params['Variable'] == variable]['Discount Factor']
                    dynamic_comps[variable], feature_dict[variable] = create_dynamic_comp(transformed_data[variable], variable, discount_factor)

            # Creating the base component
            # Input for the global discount factor
            with st.form(key='base_component'):
                discount_factor = st.number_input('Base Discount Factor', value=0.9999, step=0.0001, format="%.4f")
                submit_button = st.form_submit_button(label='Run Regression')
                st.markdown(
                """
                <style>
                .css-18e3th9 {
                    flex: 1;
                    width: 100%;
                    max-width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
                )

                # After form submission
                if submit_button:
                    # discount_factor = 0.9999 #set the base discount factor
                    base_component = trend(degree = 0, discount = discount_factor, name='intercept')
        else:
            st.write('### Please select at least 1 variable in model.')
            
        
    st.write("Evaluating performance metrics ...")
    time.sleep(sleep_time)
    
    
    st.write("Displaying performance metrics ...")
    time.sleep(sleep_time)

    
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
