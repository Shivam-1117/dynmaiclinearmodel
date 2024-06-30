import streamlit as st
from io import BytesIO
import xlsxwriter
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.optimize import minimize
from pydlm import dlm, trend, dynamic
from itertools import product, combinations


def submitted():
    st.session_state.submitted = True
def reset():
    st.session_state.submitted = False

# Lag and Adstock Functions
def apply_lag(media_data, lag = 0):
  return media_data.shift(lag, fill_value = 0)

def apply_adstock(media_data, decay = 0):
  media_data = media_data.to_numpy()
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

def get_modelResults(model, variables, model_data, actual, period):
  coefficients = pd.DataFrame()
  avp = pd.DataFrame()
  contributions = pd.DataFrame()
  coefficients['Period'] = period
  avp['Period'] = period
  contributions['Period'] = period
  avp['Actual'] = actual
  coefficients['Base'] = np.array(model.getLatentState(filterType='forwardFilter', name = 'intercept')).flatten()
  contributions['Base'] = coefficients['Base']
  for variable in variables:
    coefficients[variable] = np.array(model.getLatentState(filterType='forwardFilter', name = variable)).flatten()
    contributions[variable] = coefficients[variable] * model_data[variable]
  avp['Predicted'] = contributions.iloc[:, 1:].sum(axis = 1)
  r2 = round(r2_score(avp['Actual'], avp['Predicted']), 2)
  mape = np.mean(np.abs((avp['Actual'] - avp['Predicted']) / avp['Actual'])) * 100
  if model_data.shape[1] == 1:
    vif = pd.DataFrame({'Variable': model_data.columns,
                        'VIF': [np.nan]})
  else:
    vif = pd.DataFrame({'Variable': model_data.columns,
                        'VIF': [round(variance_inflation_factor(model_data.values, i), 2) for i in range(model_data.shape[1])]})
  vif['Variable'] = vif['Variable'].str.split('_').str[0]
  model_stats = pd.DataFrame({'R2': [r2],
                              'MAPE': [mape]})
  residuals = avp['Actual'] - avp['Predicted']
  n = len(residuals)
  k = model_data.shape[1]
  mean_residuals = np.mean(residuals)
  standard_error_residuals = np.sqrt(np.sum((residuals - mean_residuals)**2) / (n - k - 1))

  model_data_mean = model_data.mean(axis = 0)
  standard_error_model_data = np.sqrt(np.sum((model_data - model_data_mean)**2) / (n - k - 1))
  standard_error = pd.DataFrame(standard_error_model_data/standard_error_residuals).reset_index().rename(columns = {'index': 'Variable',
                                                                                                0: 'SE'})

  coefficients_mean = pd.DataFrame(coefficients.mean(axis = 0)).reset_index().rename(columns = {'index': 'Variable',
                                                                                                0: 'Coeff_Mean'})
  temp = coefficients_mean.merge(standard_error, on = 'Variable')
  tstats = pd.DataFrame()
  tstats['Variable'] = temp['Variable']
  tstats['tstat'] = pd.DataFrame(temp['Coeff_Mean']/temp['SE'])
  tstats['Variable'] = tstats['Variable'].str.split('_').str[0]
  degrees_of_freedom = n - k
  p_values = []
  for i in range(tstats.shape[0]):
    tstat = tstats.loc[i, 'tstat']
    p_values.append(2 * (1 - stats.t.cdf(np.abs(tstat), degrees_of_freedom)))
  tstats['pvalue'] = p_values
  neg_coeff_count = (coefficients.iloc[:, 1:] < 0).sum(axis = 0)
  column_sums = contributions.iloc[:, 1:].sum(axis=0)
  total_sum = column_sums.sum()
  percentage_contributions = (column_sums / total_sum) * 100
  percentage_contributions = percentage_contributions.reset_index().rename(columns={'index': 'Variable', 0: 'Contribution %'})
  percentage_contributions['Contribution %'] = percentage_contributions['Contribution %'].apply(lambda x: round(x, 2))
  variable_stats = pd.DataFrame(neg_coeff_count).reset_index().rename(columns={'index': 'Variable', 0: 'Neg Coeff Count'})
  variable_stats = pd.merge(variable_stats, percentage_contributions, on='Variable', how='left')
  variable_stats['Variable'] = variable_stats['Variable'].str.split('_').str[0]
  variable_stats = pd.merge(variable_stats, tstats, on='Variable', how='left')
  variable_stats = pd.merge(variable_stats, vif, on='Variable', how='left')
  return coefficients, avp, contributions, model_stats, variable_stats


def objective_function(x, original_contrib, model_data):
  d, g = x
  M = max(original_contrib)
  adbug_contrb = (M * (model_data**g)) / (d + model_data**g)
  return np.sum((original_contrib - adbug_contrb)**2)

def generate_response_curves(original_contrib, model_data):
  x0 = np.array([1, 1])
  min_results = minimize(objective_function, x0, args=(original_contrib, model_data), method='nelder-mead',
                 options={'xatol': 1e-8, 'disp': True})
  return [max(original_contrib), min_results.x[0], min_results.x[1]]

def plot_response_curves(curve_params):
  response_curve_data = pd.DataFrame()
  M, D, G, avg_op_level, cf = curve_params['M'], curve_params['D'], curve_params['G'], curve_params['Avg Op Level'], curve_params['Coversion Factor']
  scale = curve_params['scale']
  cprp, price = curve_params['cprp'], curve_params['price']
  adstock_difference = avg_op_level/scale
  adstock_data = [0]*70
  for i in range(len(adstock_data)):
    if i == 0:
      adstock_data[i] = 0
    else:
      adstock_data[i] = adstock_data[i-1] + adstock_difference
  response_curve_data['adstock_data'] = adstock_data
  response_curve_data['activity'] = response_curve_data['adstock_data'] * cf
  adbug_contrib = (M * (response_curve_data['adstock_data']**G)) / (D + response_curve_data['adstock_data']**G)
  response_curve_data['adbug_contrib'] = adbug_contrib
  response_curve_data['spends'] = response_curve_data['activity'] * cprp
  response_curve_data['revenue'] = response_curve_data['adbug_contrib'] * price
  response_curve_data['roi'] = response_curve_data['revenue'] * response_curve_data['spends']
  max_adbug_contrib = max(response_curve_data['adbug_contrib'])
  response_curve_data['sat_level'] = response_curve_data['adbug_contrib'] / max_adbug_contrib*100
  current_avg = response_curve_data[response_curve_data['activity'] >= avg_op_level].index[0]
  current_avg_point = response_curve_data.iloc[current_avg]
  breakthrough = response_curve_data[response_curve_data['sat_level'] >= 10].index[0]
  breakthrough_point = response_curve_data.iloc[breakthrough]
  start_sat = response_curve_data[response_curve_data['sat_level'] >= 90].index[0]
  start_sat_point = response_curve_data.iloc[start_sat]
  full_sat = response_curve_data[response_curve_data['sat_level'] >= 95].index[0]
  full_sat_point = response_curve_data.iloc[full_sat]
  return response_curve_data, current_avg_point, breakthrough_point, start_sat_point, full_sat_point

def get_simulated_data(last_year, percentage_change):
  last_year = last_year.to_numpy()
  total_last_year = sum(last_year)
  weights = last_year/total_last_year
  new_total_last_year = total_last_year*(1 + (percentage_change/100))
  new_last_year = weights * new_total_last_year
  return new_last_year



# Page title
st.set_page_config(layout='wide', page_title='Dynamic Linear Modelling App', page_icon='🏗️')
st.title('🏗️ Dynamic Linear Modelling App')
sleep_time = 1

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

def regression_section():
    st.session_state.regression_section = True
    st.header('1. Import Raw Data')
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

        if 'model_config_submitted' not in st.session_state:
            st.session_state.model_config_submitted = False        

        # if not st.session_state.model_config_submitted:
        with st.form(key='model_config'):
            for var in X.columns:
                input_cols = st.columns(9)
                input_cols[0].write(f"**{var}**")
                in_model = input_cols[1].selectbox('', to_test_options, key=f'{var}_in_model')
                vtype = input_cols[2].selectbox('', var_type, key=f'{var}_vtype')
                lag_min = input_cols[3].selectbox('', lag_options, key=f'{var}_lag_min')
                lag_max = input_cols[4].selectbox('', lag_options, key=f'{var}_lag_max')
                decay_steps = input_cols[5].text_input('', value=1, key=f'{var}_decay_steps')
                decay_min = input_cols[6].text_input('', value=1.00, key=f'{var}_decay_min')
                decay_max = input_cols[7].text_input('', value=1.00, key=f'{var}_decay_max')
                discount_factor = input_cols[8].text_input('', value=1.0000, key=f'{var}_discount_factor')
                user_inputs[var] = {
                    'In Model': in_model,
                    'Variable Type': vtype,
                    'Lag Min': lag_min,
                    'Lag Max': lag_max,
                    'Decay Steps': int(decay_steps),
                    'Decay Min': round(float(decay_min), 2),
                    'Decay Max': round(float(decay_max), 2),
                    'Discount Factor': min(round(float(discount_factor), 4), 0.9999)
                }
            # Submit button
            submit_button_config = st.form_submit_button(label='Submit')
        # When the form is submitted
        if submit_button_config:
            st.session_state.model_config_submitted = True
            # Convert the user inputs dictionary to a DataFrame
            with st.status("Running ...", expanded=True) as status:
                st.write("Setting up the model configuration ...")
                time.sleep(sleep_time)
            model_params = pd.DataFrame.from_dict(user_inputs, orient='index').reset_index().rename(columns = {'index': 'Variable'})
            if 'In Model' in model_params['In Model'].tolist() or 'Outside Model' in model_params['In Model'].tolist():
                st.write('### Regression Model Parameters (required for simulator)')
                st.dataframe(model_params)
                st.session_state['model_params'] = model_params
            else:
                st.write('### Please select at least 1 variable in model.')

        if st.session_state.model_config_submitted and 'model_params' in st.session_state.keys():
                # Creating the transformed dataset
                model_params = st.session_state['model_params']
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
                        if step == 0:
                            decays = [model_params.loc[i, 'Decay Min']]
                        else:
                            decays = np.arange(model_params.loc[i, 'Decay Min'], model_params.loc[i, 'Decay Max'], step)
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
                with st.form(key='base_component'):
                    discount_factor = st.text_input('Base Discount Factor', value='0.9999')
                    submit_button_reg = st.form_submit_button(label='Run Regression', on_click = submitted)
                if submit_button_reg:
                    base_component = trend(degree = 0, discount = min(round(float(discount_factor), 4), 0.9999), name='intercept')
                    models_df = pd.DataFrame()
                    id = 0
                    if len(outside_vars_ids.keys()) == 0:
                        model_id = 'model_' + str(id)
                        id += 1
                        lags = []
                        decays = []
                        for variable in not_outside_vars:
                            lags.append(model_params.loc[model_params['Variable'] == variable, 'Lag Min'].tolist()[0])
                            decays.append(model_params[model_params['Variable'] == variable]['Decay Min'].tolist()[0])

                        models_df = pd.DataFrame({'model_id': [model_id for i in range(len(not_outside_vars))],
                                    'outside_variables': not_outside_vars,
                                    'outside_lag': lags,
                                    'outside_decay': decays
                                    })
                    else:    
                        for r in range(1, len(outside_vars_ids.keys())+ 1):  # generate combinations of size 1, 2, and 3
                            for combination in combinations(outside_vars_ids.keys(), r):
                                product_combinations = list(product(*(outside_vars_ids[key] for key in combination)))
                                for item in product_combinations:
                                    model_id = 'model_' + str(id)
                                    outside_vars = [key for key in combination]
                                    id += 1
                                    lags = []
                                    decays = []
                                    for i in range(len(item)):
                                        lags.append(outside_vars_dict[outside_vars[i]][item[i]][1])
                                        decays.append(outside_vars_dict[outside_vars[i]][item[i]][2])

                                    for variable in not_outside_vars:
                                        outside_vars.append(variable)
                                        lags.append(model_params.loc[model_params['Variable'] == variable, 'Lag Min'].tolist()[0])
                                        decays.append(model_params[model_params['Variable'] == variable]['Decay Min'].tolist()[0])

                                    new_row = pd.DataFrame({'model_id': [model_id for i in range(len(outside_vars))],
                                                'outside_variables': outside_vars,
                                                'outside_lag': lags,
                                                'outside_decay': decays
                                                })
                                    models_df = pd.concat([models_df, new_row], ignore_index = True)
                        
                    all_models = models_df['model_id'].unique()
                    model_stats_all = pd.DataFrame()
                    variable_stats_all = pd.DataFrame()

                    for i in range(len(all_models)):
                        model_df = models_df[models_df['model_id'] == all_models[i]].reset_index(drop = True)
                        model = dlm(y)
                        model.add(base_component)
                        list_of_model_vars = []
                        for j in range(len(model_df)):
                            variable = model_df.loc[j, 'outside_variables']
                            lag = model_df.loc[j, 'outside_lag']
                            decay = model_df.loc[j, 'outside_decay']
                            variable_new = variable + '_' + str(lag) + '_' + str(decay)

                            if variable in list(outside_vars_dict.keys()):
                                model.add(dynamic_comps[variable_new])
                                list_of_model_vars.append(variable_new)
                            else:
                                model.add(dynamic_comps[variable])
                                list_of_model_vars.append(variable)
                        model.fit()
                        coefficients, avp, contributions, model_stats, variable_stats = get_modelResults(model, list_of_model_vars,
                                                                                                        model_data = transformed_data[list_of_model_vars],
                                                                                                        actual = y,
                                                                                                        period = retail_data['Timeframe'])

                        model_stats.insert(loc=0, column='model_id', value=all_models[i])
                        variable_stats.insert(loc=0, column='model_id', value=all_models[i])

                        model_stats_all = pd.concat([model_stats_all, model_stats], ignore_index = True)
                        variable_stats_all = pd.concat([variable_stats_all, variable_stats], ignore_index = True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('### Model Results')
                        st.dataframe(model_stats_all)
                    with col2:
                        st.write('### Variable Stats')
                        st.dataframe(variable_stats_all)

                    st.write('### Actual Vs Predicted')
                    fig, ax = plt.subplots()
                    color = 'tab:blue'
                    ax.set_xlabel('Period')
                    ax.set_ylabel('KPI', color=color)
                    ax.plot(avp['Period'], avp['Actual'], color='blue')
                    ax.plot(avp['Period'], avp['Predicted'], color='orange')
                    ax.tick_params(axis = 'y', labelcolor=color)
                    ax.legend()
                    fig.tight_layout()
                    col1, col2 = st.columns(vertical_alignment="top", spec = [0.6, 0.4])
                    
                    with col1:
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(avp, height=560, width=600)
                    
                    col1, col2, col3 = st.columns(vertical_alignment="top", spec = [0.33, 0.33, 0.33])
                    
                    with col1:
                        st.write('### Transformed Data (required for response curves)')
                        st.dataframe(transformed_data)
                    with col2:
                        st.write('### Coefficients')
                        st.dataframe(coefficients)
                    with col3:
                        st.write('### Contributions (required for response curves)')
                        st.dataframe(contributions)

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        model_params.to_excel(writer, sheet_name='Model Parameters', index=False)
                        avp.to_excel(writer, sheet_name='AVP', index=False)
                        model_stats_all.to_excel(writer, sheet_name='Model Stats', index=False)
                        variable_stats_all.to_excel(writer, sheet_name='Variable Stats', index=False)
                        transformed_data.to_excel(writer, sheet_name='Transformed Data', index=False)
                        coefficients.to_excel(writer, sheet_name='Coefficients', index=False)
                        contributions.to_excel(writer, sheet_name='Contributions', index=False)
                        retail_data.to_excel(writer, sheet_name='Raw Data', index=False)

                    output.seek(0)
                    st.download_button(
                        label="Download Model Dump",
                        data = output,
                        file_name='Model_Dump.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

def response_curves_section():
    st.session_state.response_curves_section = True
    # Response Curves
    st.header('1. Import Model Dump')
    uploaded_file = st.file_uploader("Upload Model Dump", type=["xlsx"])
    if uploaded_file is not None:
        sheets_dict = pd.read_excel(uploaded_file, sheet_name=None)
        # Display each sheet's dataframe
        for sheet_name, df in sheets_dict.items():
            if sheet_name == 'Raw Data':
                raw_data = df.copy()
            if sheet_name == 'Transformed Data':
                model_data = df.copy()
            if sheet_name == 'Contributions':
                contributions = df.copy()
    # Initiate the model building process
    if uploaded_file:
        with st.status("Running ...", expanded=True) as status:
            st.write("Uploading data ...")
            time.sleep(sleep_time)
        response_curve_params = pd.DataFrame()
        for variable in model_data.columns:
            avg_op_level = raw_data.loc[raw_data[variable] != 0, variable].mean()
            cf = sum(raw_data[variable])/sum(model_data[variable])
            mdg = generate_response_curves(contributions[variable], model_data[variable])
            mdg.append(avg_op_level)
            mdg.append(cf)
            response_curve_params[variable] = mdg
        response_curve_params = response_curve_params.transpose()
        response_curve_params.columns = ['M', 'D', 'G', 'Avg Op Level', 'Coversion Factor']
        response_curve_params = response_curve_params.reset_index().rename(columns = {'index': 'Variable'})
        st.write('### Resposne Curve Parameters (Requiredfor Simulator)')
        st.dataframe(response_curve_params)
        variable = st.selectbox('Select the variable name:', response_curve_params['Variable'].tolist())
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            response_curve_params.to_excel(writer, sheet_name='RC Parameters', index=False)

        if variable:
            M = response_curve_params.loc[response_curve_params['Variable'] == variable, 'M'].tolist()[0]
            D = response_curve_params.loc[response_curve_params['Variable'] == variable, 'D'].tolist()[0]
            G = response_curve_params.loc[response_curve_params['Variable'] == variable, 'G'].tolist()[0]
             # Arrange the dropdown and input fields in a single row
            col1, col2, col3= st.columns([1, 1, 1])
            with col1:
                st.write('M')
                st.write(M)
            with col2:
                st.write('D')
                st.write(D)
            with col3:
                st.write('G')
                st.write(G)
             # Arrange the dropdown and input fields in a single row
            col1, col2, col3= st.columns([1, 1, 1])
            with col1:
                cprp = st.text_input('Enter CPRP:', value=1.0)
            with col2:
                price = st.text_input('Enter Price:', value=1.0)
            with col3:
                scale = st.text_input('Enter Scale:', value=10)

            if st.button('Generate Response Curve'):
                curve_params = {'M': M,
                                'D': D,
                                'G': G,
                                'Avg Op Level': response_curve_params.loc[response_curve_params['Variable'] == variable, 'Avg Op Level'].tolist()[0],
                                'Coversion Factor': response_curve_params.loc[response_curve_params['Variable'] == variable, 'Coversion Factor'].tolist()[0],
                                'cprp': float(cprp),
                                'price': float(price),
                                'scale': float(scale)
                                }

                response_curve_data, current_avg_point, breakthrough_point, start_sat_point, full_sat_point = plot_response_curves(curve_params)
                fig, ax = plt.subplots()
                color = 'tab:blue'
                ax.set_xlabel('Activity')
                ax.set_ylabel('Adbug Contribution', color=color)
                ax.plot(response_curve_data['activity'], response_curve_data['adbug_contrib'], color=color)
                ax.scatter(current_avg_point['activity'], current_avg_point['adbug_contrib'], color='green', label='Current Average')
                ax.scatter(breakthrough_point['activity'], breakthrough_point['adbug_contrib'], color='black', label='Breakthrough')
                ax.scatter(start_sat_point['activity'], start_sat_point['adbug_contrib'], color='orange', label='Start of Saturation')
                ax.scatter(full_sat_point['activity'], full_sat_point['adbug_contrib'], color='red', label='Full Saturation')
                ax.tick_params(axis='y', labelcolor=color)
                ax.set_title('Response Curve for ' + variable)
                ax.legend()
                fig.tight_layout()
                col1, col2 = st.columns(spec = [0.6, 0.4])
                with col1:
                    st.pyplot(fig)
                with col2:
                    st.dataframe(response_curve_data, height=560)

                with pd.ExcelWriter(output, engine='openpyxl', if_sheet_exists='replace', mode = 'a') as writer:
                    response_curve_data.to_excel(writer, sheet_name=variable, index=False)
        output.seek(0)
        st.download_button(
            label="Download Resposne Curves Data",
            data = output,
            file_name='Resposne Curves Data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

def simulator_section():
    st.session_state.simulator_section = True
    # Simulator
    st.header('1. Import Model Dump')
    uploaded_file = st.file_uploader("Upload Model Dump", type=["xlsx"])
    if uploaded_file is not None:
        sheets_dict = pd.read_excel(uploaded_file, sheet_name=None)
        for sheet_name, df in sheets_dict.items():
            if sheet_name == 'Raw Data':
                raw_data = df.copy()
            if sheet_name == 'Model Parameters':
                model_params = df.copy()

    st.header('2. Import Response Curve Data File')
    uploaded_file = st.file_uploader("Upload Resposne Curves Data", type=["xlsx"])
    if uploaded_file is not None:
        sheets_dict = pd.read_excel(uploaded_file, sheet_name=None)
        for sheet_name, df in sheets_dict.items():
            if sheet_name == 'RC Parameters':
                response_curve_params = df.copy()
    
    if uploaded_file:
        with st.status("Running ...", expanded=True) as status:
            st.write("Uploading data ...")
            time.sleep(sleep_time)
        model_params = model_params.merge(response_curve_params, on = 'Variable', how = 'left')
        transformed_data_new = pd.DataFrame()
        next_year_contribs = pd.DataFrame()
        percentage = 10
        for variable in raw_data.columns[2:]:
            transformed_data_new[variable] = get_simulated_data(raw_data[variable][len(raw_data) - 12:], percentage_change = 10)
            lag = model_params[model_params['Variable'] == variable]['Lag Min'].tolist()[0]
            decay = model_params[model_params['Variable'] == variable]['Decay Min'].tolist()[0]
            M = model_params[model_params['Variable'] == variable]['M'].tolist()[0]
            D = model_params[model_params['Variable'] == variable]['D'].tolist()[0]
            G = model_params[model_params['Variable'] == variable]['G'].tolist()[0]
            transformed_data_new[variable] = apply_lag(transformed_data_new[variable], lag)
            transformed_data_new[variable] = apply_adstock(transformed_data_new[variable], decay)
            next_year_contribs[variable] = (M * (transformed_data_new[variable] ** G)) / (D + transformed_data_new[variable] ** G)

if 'page_regression' not in st.session_state:
            st.session_state.page_regression = False
if 'page_response_curves' not in st.session_state:
            st.session_state.page_response_curves = False 
if 'page_simulator' not in st.session_state:
            st.session_state.page_simulator = False 
if 'regression_section' not in st.session_state:
            st.session_state.regression_section = False
if 'response_curves_section' not in st.session_state:
            st.session_state.response_curves_section = False
if 'simulator_section' not in st.session_state:
            st.session_state.simulator_section = False

page_regression = st.sidebar.button("Regression")
page_response_curves = st.sidebar.button("Response Curves")
page_simulator = st.sidebar.button("Simulator")

if page_regression and not st.session_state.regression_section:
    st.session_state.page_regression = True
    st.session_state.page_response_curves = False
    st.session_state.response_curves_section = False
    st.session_state.page_simulator = False
    st.session_state.simulator_section = False
    st.title("Run Regression")
    regression_section()
    st.sidebar.empty()
elif st.session_state.regression_section and not page_response_curves and not page_simulator:
    st.sidebar.empty()
    regression_section()
elif page_response_curves and not st.session_state.response_curves_section:
    st.session_state.page_regression = False
    st.session_state.regression_section = False
    st.session_state.page_response_curves = True
    st.session_state.page_simulator = False
    st.session_state.simulator_section = False
    st.title("Generate Response Curves")
    response_curves_section()
elif st.session_state.response_curves_section and not page_regression and not page_simulator:
    response_curves_section()
elif page_simulator and not st.session_state.simulator_section:
    st.session_state.page_regression = False
    st.session_state.regression_section = False
    st.session_state.page_response_curves = False
    st.session_state.response_curves_section = False
    st.session_state.page_simulator = True
    st.title("Simulator")
    simulator_section()
elif st.session_state.simulator_section and not page_regression and not page_response_curves:
    simulator_section()
    
if not page_regression and not page_response_curves and not page_simulator:
        st.session_state.page_regression = False
        st.session_state.page_response_curves = False
        st.session_state.page_simulator = False