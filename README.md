# 🏗️ ML model builder template

A simple Streamlit app that lets you build simple ML models with scikit-learn. 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-model-builder-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

Certainly! Here’s a well-formatted version of the information for the README section of your Git repository:

---

## What can this app do?

Welcome to our dynamic linear modeling app! Here, you can effortlessly build and refine your models from start to finish. Dive into the data, generate insightful response curves for your confirmed models, and explore media spend simulations to optimize your marketing strategy.

## How to use the app?

🚀 Welcome to our app with three powerful modules!

🔹 **Regression**: Build and refine your models with ease.

🔹 **Response Curves**: Visualize and analyze how changes in media spend impact your results.

🔹 **Simulator**: Experiment with different media spend scenarios to find the optimal strategy.

To get started, head over to the sidebar and select the task you want to perform. Use the intuitive flowcharts to guide you through each module step by step.

### Regression Module Workflow

1. **Import the Raw Data file**:
   - Start by uploading your dataset.

2. **Set the Model Configuration**:
   - Configure your model parameters like lag and decay.
   - Click on the 'Submit' button.

3. **Select the Base Discount Factor**:
   - Choose your discount factor and click on 'Run Regression'.

4. **Download the Model Dump**:
   - After running the regression, click on the 'Download Model Dump' button below the Actual vs Predicted plot to save your model.

### Response Curves Module Workflow

1. **Import the Model Dump**:
   - Start by uploading your saved model dump.

2. **Select Variable Name from the Dropdown**:
   - Choose the variable you want to analyze.

3. **Enter CPRP, Price, and Scale**:
   - Input the required values for CPRP, price, and scale.

4. **Generate Response Curve**:
   - Click the 'Generate Response Curve' button to visualize the response.

5. **Finalize and Confirm the Curve**:
   - Adjust the scale values if necessary.
   - Click 'Confirm Curve' and then select another variable to repeat the process.

6. **Download Response Curve Data**:
   - Once all curves are generated and confirmed, click 'Download Response Curve Data' to save your work.

### Simulator Module Workflow

1. **Import the Model Dump and Response Curve Data File**:
   - Start by uploading your saved model dump and response curve data file.

2. **Select Percentage Change in Last Year's Media Spend**:
   - Use the slider to set the percentage change in media spends over the last year.

3. **Simulate**:
   - Click the 'Simulate' button to generate the simulated data.