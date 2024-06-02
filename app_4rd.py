import streamlit as st
import pandas as pd
from io import BytesIO
import joblib
import openpyxl
import xgboost
import sklearn
import imblearn
# Load pre-trained model and input features
Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

# Define the prediction function
def prediction(Customer_Segment, Bill_Cycle, Current_Status, Live_Contracts,
               Hotline_Count_Last_6_months, ADSL, Activation_source,
               Tarrif_combinetion, Standardized_Payment_Channel,
               variance_days, AVG_payment_amount, activation_years,
               activation_months, activation_day, Bill_Cycle_Customer_Segment,
               Tariff_Model_str):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0, "Customer Segment"] = Customer_Segment
    test_df.at[0, "Bill Cycle"] = Bill_Cycle
    test_df.at[0, "Current Status"] = Current_Status
    test_df.at[0, "Live Contracts"] = Live_Contracts
    test_df.at[0, "Hotline Count (Last 6 months)"] = Hotline_Count_Last_6_months
    test_df.at[0, "ADSL"] = ADSL
    test_df.at[0, "Activation source"] = Activation_source
    test_df.at[0, "Tarrif_combinetion"] = Tarrif_combinetion
    test_df.at[0, "Standardized Payment Channel"] = Standardized_Payment_Channel
    test_df.at[0, "variance_days"] = variance_days
    test_df.at[0, "AVG_payment_amount"] = AVG_payment_amount
    test_df.at[0, "activation_years"] = activation_years
    test_df.at[0, "activation_months"] = activation_months
    test_df.at[0, "activation_day"] = activation_day
    test_df.at[0, "Bill_Cycle_Customer_Segment"] = Bill_Cycle_Customer_Segment
    test_df.at[0, "Tariff_Model_str"] = Tariff_Model_str
    result = Model.predict(test_df)[0]
    return result

# Define the class mapping for the target column
class_mapping = {0: 'outstanding', 1: 'Normal', 2: 'risky', 3: 'very risky'}

# Define the Streamlit app
def main():
    st.title("Hotline Action Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        # Read the uploaded file
        input_df = pd.read_excel(uploaded_file, dtype=str)
        st.write("Uploaded Data")
        st.dataframe(input_df)

        # Make predictions
        predictions = []
        for index, row in input_df.iterrows():
            pred = prediction(row['Customer Segment'], row['Bill Cycle'], row['Current Status'], row['Live Contracts'],
                              row['Hotline Count (Last 6 months)'], row['ADSL'], row['Activation source'],
                              row['Tarrif_combinetion'], row['Standardized Payment Channel'], row['variance_days'],
                              row['AVG_payment_amount'], row['activation_years'], row['activation_months'],
                              row['activation_day'], row['Bill_Cycle_Customer_Segment'], row['Tariff_Model_str'])
            predictions.append(pred)
        
        # Map the numeric predictions to their respective classes
        input_df['Hotline_class'] = [class_mapping[pred] for pred in predictions]

        # Display the data with predictions
        st.write("Data with Predictions")
        st.dataframe(input_df)

        # Function to convert dataframe to excel
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        # Convert dataframe to excel
        processed_data = to_excel(input_df)

        # Download button
        st.download_button(
            label="Download Excel",
            data=processed_data,
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
