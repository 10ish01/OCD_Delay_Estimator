import streamlit as st
import pandas as pd
import numpy as np
import joblib
import csv
import os
from PIL import Image
from datetime import datetime, date, time
import subprocess
import shutil
# === Load Models ===
classifier = joblib.load("model/classifier.pkl")
regressor = joblib.load("model/regressor.pkl")

# === Load Test Metadata ===
test_data = pd.read_csv("data/test_data.csv")

# === Page Setup ===
st.set_page_config(layout="wide", page_title="OCD Delay Predictor")
st.sidebar.image("assets/logo.png", use_container_width=True)
st.title("OCD Delay Predictor")

# === Sidebar Menu ===
st.sidebar.markdown("### Navigation")
nav_option = st.sidebar.radio("Select Page", ["Predict Delay", "Simple Estimator", "Update Data", "Help"])

# === Sidebar: Breakdown Form ===
st.sidebar.markdown("---")
st.sidebar.markdown("### Report a Breakdown")
with st.sidebar.form("breakdown_form"):
    machine_option = st.selectbox("Select Machine", options=[f"Vitros {i}" for i in range(1, 7)] + ["Other"])
    machine_name = st.text_input("Enter Machine Name") if machine_option == "Other" else machine_option
    breakdown_date = st.date_input("Breakdown Date", value=date.today())
    breakdown_time = st.time_input("Breakdown Time", value=datetime.now().time())
    duration = st.number_input("Duration (minutes)", min_value=0, max_value=1440, value=0)
    description = st.text_area("Description")
    submit = st.form_submit_button("Submit Breakdown")

    if submit:
        breakdown_datetime = datetime.combine(breakdown_date, breakdown_time)
        breakdown_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_name": machine_name,
            "breakdown_datetime": breakdown_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_mins": duration,
            "description": description
        }
        file_path = "data/breakdowns.csv"
        os.makedirs("data", exist_ok=True)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=breakdown_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(breakdown_data)
        st.sidebar.success(f"Breakdown reported")

# === Main Page ===
if nav_option == "Predict Delay":
    st.markdown("---")
    st.subheader("Test Entry")
    col1, col2, col3 = st.columns(3)

    with col1:
        test_name = st.selectbox("Select Test Name", test_data["test_name"].unique())
        test_row = test_data[test_data["test_name"] == test_name].iloc[0]
        avg_test_time = float(test_row["avg_test_time"])
        
        
        
        avg_delay_when_breakdown = float(test_row["avg_delay_when_breakdown"])
        tests_in_sample = st.number_input("Tests in Same Sample", min_value=1, value=4)
        lab_tat = st.number_input("Test Lab TAT", value= int(test_row["lab_tat"]))
        
        
        
      


    with col2:
        selected_date = st.date_input("Test Date", value=date.today())
        selected_time = st.time_input("Sample Arrival Time")
        selected_datetime = datetime.combine(selected_date, selected_time)
        day = selected_datetime.weekday()
        hour = selected_time.hour
        if hour < 7: time_bin = 0
        elif hour < 9: time_bin = 1
        elif hour < 11: time_bin = 2
        elif hour < 13: time_bin = 3
        elif hour < 16: time_bin = 4
        elif hour < 20: time_bin = 5
        else: time_bin = 6
        breakdown = st.selectbox("Any Breakdown Before Test Arrival?", ["No", "Yes"]) == "Yes"

        breakdown_duration = 0

           
        if not breakdown:
            breakdown_class = 0
        elif breakdown_duration <= 30:
            breakdown_class = 1
        elif breakdown_duration <= 60:
            breakdown_class = 2
        elif breakdown_duration <= 90:
            breakdown_class = 3
        else:
            breakdown_class = 4


    with col3:
        cumulative_delays = st.number_input("Cumulative Delayed Tests Before", min_value=0, value=0,step=5)
        current_processing = st.number_input("Current Processing Load", min_value=0, value=0,step=5)
        if breakdown:
           
             breakdown_duration = st.number_input("Breakdown Duration (mins)", min_value=1, value=30,step=5)
    
    col4, col5, col6 = st.columns(3)
    with col4:
      avg_test_delay = float(test_row["avg_test_delay"])
      st.text(f"Average Test Time: {avg_test_time:.2f} hr")
    with col5:
        delay_ratio = float(test_row["delay_ratio"])
        st.text(f"Delay Percentage For The Test: {(delay_ratio)*100:.2f}%") 
    with col6:
        avg_test_delay = float(test_row["avg_test_delay"])
        st.text(f"Average Delay When This Test Gets Delayed: {avg_test_delay:.2f} hr")
    if st.button("Predict Delay"):
        breakdown_flag = 1 if breakdown else 0
        input_df = pd.DataFrame([{
    "lab_tat": lab_tat,
    "day": day,
    "time_bin": time_bin,
    "cumulative_delays": cumulative_delays,
    "current_processing": current_processing,
    "avg_test_time": avg_test_time,
    "tests_in_sample": tests_in_sample,
    "delay_ratio": delay_ratio,
    "avg_test_delay": avg_test_delay,
    "breakdown_flag": int(breakdown_flag),
    "breakdown_class": breakdown_class,
    "avg_delay_when_breakdown": avg_delay_when_breakdown
}])

        try:
            prob = classifier.predict_proba(input_df)[0, 1]
            delay_class = int(prob >= 0.4)
            delay = regressor.predict(input_df)[0]
            delay = round(min(max(delay, 0), 4.5), 2)
            hours, minutes = divmod(int(delay * 60), 60)
            if delay_class == 1:

             st.success(f"Predicted Delay: **{hours} hr {minutes} min**")
            else:
             st.success("A delay is unlikely")
             with st.expander("Force show delay estimate"):
       
                 st.write(f"Estimated Delay: **{hours} hr {minutes} min**")

        except Exception as e:
            st.error(f"Prediction error: {e}")


elif nav_option==("Simple Estimator"):
     st.header("Get a blanket maximum delay estimation")
     st.markdown(">Useful when you need a simple maximum delay amount or lack datapoints to accuratley make prediction using the ML model")

    # Default values
     default_values = {
        "avg_tests": 4.0,
        "avg_time": 1.0,
        "broken_machines": 0,
        "breakdown_time": 0.0,
        "lab_tat": 2.5
                         }

    # Input fields
     samples = st.number_input("Number of samples in machine:", min_value=1.0, value=1.0,step=1.0)
     avg_tests = st.number_input("Avg no. of tests per sample:", min_value=1.0, value=default_values["avg_tests"])
    
     avg_time = st.number_input("Avg time per test (Hr):", min_value=0.0, value=default_values["avg_time"])
    
     broken_machines = st.number_input("No. of machines in breakdown:", min_value=0, value=default_values["broken_machines"])
     breakdown_time = st.number_input("Breakdown time per machine (min):", min_value=0.0, value=default_values["breakdown_time"])
     lab_tat = st.number_input("Machine TAT (Hr):", min_value=0.0, value=default_values["lab_tat"])

     no_of_machines = 6  # fixed

         
          
     total_tests = samples * avg_tests
     total_time = total_tests * avg_time
     delay_due_to_breakdowns = (broken_machines * breakdown_time) / 60
     available_time = lab_tat * no_of_machines - delay_due_to_breakdowns
     delay_hours = max(0, (total_time - available_time) / no_of_machines)

     st.success(f"Your estimated maximum delay for a test is **{delay_hours:.2f} hours**")

# ... inside nav_option == "Update Data" section:
elif nav_option == "Update Data":


    st.header("Upload New Dataset")

# Step 1: Show previous info if exists
    with st.expander("Previous Uploaded File Info"):
   

     if os.path.exists(os.path.join("data", "data_info.txt")):
    # your code here

       with open("data_info.txt", "r") as f:
        prev_info = f.read()
       
        st.text(prev_info)
     else:
        st.info("No previous upload info found.")
    with st.expander("Expected Structure of Upload Files"):
        st.image("assets/image.png")

# Step 2: File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
     try:
        df = pd.read_csv(uploaded_file)

        # Step 3: Validate required columns
        if "sample_received_time" not in df.columns or "delay" not in df.columns:
            st.error("Required columns `sample_received_time` or `delay` are missing.")
        else:
            # Step 4: Process new file
            df['sample_received_time'] = pd.to_datetime(df['sample_received_time'], errors='coerce')
            valid_df = df.dropna(subset=['sample_received_time'])

            first_time = valid_df['sample_received_time'].min()
            last_time = valid_df['sample_received_time'].max()

            delay_count = df[df['delay'] > 0].shape[0]

            # Step 5: Prepare new info summary
            info_text = (
               
                f"First sample received time: `{first_time}`"
                f"\nLast sample received time: `{last_time}`"
                f"\nRows with delay: `{delay_count}`"
                f"\nTotal rows: `{len(df)}`"
            )

            st.markdown("New Upload Summary")
            st.markdown(info_text)

            # Step 6: Ask user for confirmation
            if st.button("Confirm and Upload New File"):
                os.makedirs("dataset", exist_ok=True)
                files = os.listdir("dataset")
                count = max(
                    [int(f.replace("dataset", "").replace(".csv", "")) for f in files if f.startswith("dataset") and f.endswith(".csv")] + [0]
                ) + 1
                filename = f"dataset{count}.csv"
                with open(os.path.join("dataset", filename), "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Update data_info.txt
                with open(os.path.join("data", "data_info.txt"), "w") as f:
                    f.write(info_text.replace("**", "").replace("`", ""))  # Strip markdown

                st.success(f"File uploaded as `{filename}` and info saved to `data_info.txt`")
     except Exception as e:
         st.error(f"Error processing the file: {e}")

         # Deletion section
    st.header("Delete Existing Data")
    if os.path.exists("dataset"):
        dataset_files = [f for f in os.listdir("dataset") if f.endswith(".csv")]
        if dataset_files:
            file_to_delete = st.selectbox("Select a file to delete", dataset_files)
            if st.button("Delete File"):
                os.remove(os.path.join("dataset", file_to_delete))
                st.warning(f"File `{file_to_delete}` deleted successfully!")
               
        else:
            st.info("No dataset files found to delete.")
    else:
        st.info("Dataset folder doesn't exist yet.")
        
    st.header("Retrain The Model")
    st.markdown("Upon uploading new data or deleting any old data, you can retrain the model here")
    


    CLASSIFIER_PATH = os.path.join("model", "classifier.pkl")
    REGRESSOR_PATH = os.path.join("model", "regressor.pkl")
    CLASSIFIER_BACKUP = os.path.join("model", "classifier_prev.pkl")
    REGRESSOR_BACKUP = os.path.join("model", "regressor_prev.pkl")


    if st.button("Retrain Model"):
    # Step 1: Backup old models if they exist
     if os.path.exists(CLASSIFIER_PATH):
        shutil.copy(CLASSIFIER_PATH, CLASSIFIER_BACKUP)
     if os.path.exists(REGRESSOR_PATH):
        shutil.copy(REGRESSOR_PATH, REGRESSOR_BACKUP)

    # Step 2: Show backup info
     st.markdown("Backup of previous model created inside the 'model' directory")
     st.markdown(f"- [classifier_prev.pkl](./{CLASSIFIER_BACKUP})")
     st.markdown(f"- [regressor_prev.pkl](./{REGRESSOR_BACKUP})")

    
     with st.spinner("Training in progress..."):
            try:
                result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Model retraining completed successfully!")
                    st.text(result.stdout)
                else:
                    st.error("Model training failed!")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"Error running training script: {e}")
    st.header("Model Metrics")
    with st.expander("View Model Performance Metrics:"):
     with open("model/model_metrics.txt", "r") as f:
       metrics_text = f.read()

       st.code(metrics_text, language="text")
    with st.expander("View Confusion Matrix For Classifier:"):
        st.image("assets/confusion_matrix.png")
    with st.expander("View Actual Vs Predicted Delay Plot For Regressor:"):
        st.image("assets/actual_vs_predicted_delay.png")
    with st.expander("View SHAP Feature Importance Plot:"):
        st.image("assets/shap_plot.png")

elif nav_option == "Help":
    st.header("Help Guide")
   

    st.markdown("""
    This tool is designed to better predict **delays** for tests done on Vitros 5600 machine set using ML models on past data.

    ---

    ### How the Prediction Works (2-Stage Methodology)

    The prediction system follows a **2-stage approach** for higher accuracy and clarity:

    #### **Stage 1 - Delay Classification**
    A **classifier model** is first used to estimate whether a delay is **likely** or **not** for a given test request, based on probability threshold that is dynamically derived by the model algorithm from the dataset.

    #### **Stage 2 - Delay Estimation**
    If a delay is likely (or if you still want an estimate), a **regressor model** is used to predict the **estimated number of delay hours**.

    ---

    ### Inputs Required


  
    - Select test type and manually input: 
        - Lab TAT (Pre-loaded)
        - Test Date & Time (to compute day of week and time bin) 
        - Number Of Sample In The Test 
        - Current Test Load  
        - Cumulative Delays  
        - Breakdown occurrence & duration""")

 
    st.subheader("Backend Model Insights")
    
    feature_data = {
    "Feature": [
        "avg_test_time",
        "lab_tat",
        "day",
        "time_bin",
        "tests_in_sample",
        "current_processing",
        "cumulative_delays",
        "delay_ratio",
        "avg_test_delay",
        "breakdown_flag",
        "breakdown_class",
        "avg_delay_when_breakdown"
    ],
    "Description": [
        "Historical average time (in hours) taken to process the test type.",
        "Lab Turnaround Time category (Trained only for tests with TAT as 3 or 4 hours).",
        "Day of the week when sample arrived (0=Monday to 6=Sunday).",
        "Time of day when sample arrived, binned into intervals.",
        "Number of tests associated with the same sample.",
        "Count of tests still being processed when this test arrived.",
        "Cumulative count of delayed tests earlier on the same date.",
        "Historical proportion of delays for this test type.",
        "Mean delay (in hours) when this test type was delayed in the past.",
        "1 if a breakdown occurred on the test date before the sample arrived, else 0.",
        "Category of breakdown duration: 0 (none), 1 (<30min), ..., 4 (>120min).",
        "Average delay for this test when breakdowns occurred."
    ]
}


    feature_df = pd.DataFrame(feature_data)


    st.subheader("Features Used in Model Training")
    st.dataframe(feature_df, use_container_width=True)
    st.markdown("""
    - The **classifier** is a binary model based on XGBoost lgorithm that evaluates the delay likelihood.
    - The **regressor** is trained using CATBoost regressor, takes the same inputs and outputs the predicted delay.
    - If `breakdown` is checked, the **breakdown class** (0 to 4) is derived from duration and included in the features.
    - The trained models can be accessed indepedently from the 'model' folder inside the main directory.
    - To train the models, the last 5 uploaded datasets are taken and combined to a single file,
      it can be accessed in the 'data/' folder by the name 'train_data.csv'
      
    ---

    ### Other Features

    #### Report Breakdowns 
    - Log machine breakdowns in a well structured manner, making it eaiser to use the breakdown data for post-analysis.
    - Entries are saved in a unique CSV file.
    
    #### Simple Estimator
    - In case you need simple delay estimation that does not rely on any past data, and gives a general estimate valid for
      all tests, you can use this tool instead. It gives an estimate based purely on mathematical calculations with minimum inputs.

   
    #### **Update Data**
    - Upload new CSV files to add new training data.
    - Delete exisitng CSV files that are used in training.
    - Retrain the model in case of data updations.
    - View model metrics and performance
     

    ---

    ### Tips
    - The predictions can only be made for tests having a Lab TAT of 3 or 4 hours. 
    - The delay predictions from the regressor are hard capped at a maximum of 4.5 hours.
    - Though the model can predict samples that **are not going to have any delay** with a high accuracy, it still provides a hard delay estimate for them regardless.
    - In rare cases, the model may predict no delay even when inputs (e.g., high load at any unusual time, say 10 PM) suggest otherwise.
      This typically happens when such combinations are absent in the training data. 
    - Breakdown and its effects aren't captured accurately by the model yet, machine downtime is taken into considertion rather than the severity, due to lack of 
      such classification, but provision has been left for this, breakdown class as a feature can be calculated alternately on severity rather than downtime in future data.
    - Delay predictions can only be made for tests appearing in selectbox for test name, in case you wish to make prediction for a new test, either feed in data regarding
      the test or add the test name and required test characteristics manually in 'test_data.csv' inside the 'data/' folder.

    ---
 
    >If you need further help, contact- Tanish Hora (iamtanishhora@gmail.com).
    """)