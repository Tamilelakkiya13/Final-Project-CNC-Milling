import streamlit as st
import pandas as pd
import pickle

# Define file paths
model_path = "model.pkl"

# Load model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model(model_path)

# Define label mappings
passed_visual_labels = {0: "Failed", 1: "Passed"}
machining_finalized_labels = {0: "Not Finalized", 1: "Finalized"}
tool_condition_labels = {0: "UnWorn", 1: "Worn "}

import base64
import os

def set_background_image_local():
    image_path = "tool animated.jpg"
    
    if not os.path.exists(image_path):
        st.error("Background image not found. Please check the file path.")
        return
    
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    ) 

# Call the function
set_background_image_local()
# Streamlit UI
st.title("CNC Machine Prediction Dashboard")

# Upload Test Dataset
uploaded_file = st.file_uploader("Upload test dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file, dtype=str)
            for col in df_test.columns:
                try:
                    df_test[col] = pd.to_numeric(df_test[col], errors='ignore')
                except:
                    pass
            if "Machining_Process" in df_test.columns:
    # Convert all values to string before applying `.str.lower()`
                df_test["Machining_Process"] = df_test["Machining_Process"].astype(str).str.lower()

                # Define mapping for all unique values
                machining_process_map = {
                    "starting": 0,
                    "prep": 1,
                    "layer 1 up": 2,
                    "layer 1 down": 3,
                    "repositioning": 4,
                    "layer 2 up": 5,
                    "layer 2 down": 6,
                    "layer 3 up": 7,
                    "layer 3 down": 8,
                    "end": 9
                }

                # Apply mapping and handle unknown values
                df_test["Machining_Process"] = df_test["Machining_Process"].map(machining_process_map)
                df_test["Machining_Process"] = df_test["Machining_Process"].fillna(-1).astype(int)  # Set unknown values to -1

            else:
                st.write("'Machining_Process' column not found in uploaded dataset.")    
        except:
            pass
else:
    st.write(f"Uploaded file is not matching with the model features please upload the valid test file")

# Handle "Machining_Process" column properly


# Prediction buttons
if st.button("Predict Passed Visual Inspection"):
    pred_passed_visual = model.predict(df_test)[:, 0]  # First column
    #pred_passed_visual = pred_passed_visual.round().astype(int)  # Convert to 0/1
    #pred_passed_visual = [passed_visual_labels.get(val, "Unknown") for val in pred_passed_visual]
    st.write("Passed Visual Inspection Predictions:")
    st.write("Inspection Passed" if pred_passed_visual > 0.5 else "Inspection Failed")

if st.button("Predict Machining Finalized"):
    pred_machining_finalized = model.predict(df_test)[:, 1]  # Second column
    pred_machining_finalized = pred_machining_finalized.round().astype(int)
    pred_machining_finalized = [machining_finalized_labels.get(val, "Unknown") for val in pred_machining_finalized]
    st.write("Machining Finalized Predictions:")
    st.write(pred_machining_finalized)

if st.button("Predict Tool Condition"):
    pred_tool_condition = model.predict(df_test)[:, 2]  # Third column
    pred_tool_condition = pred_tool_condition.round().astype(int)
    pred_tool_condition = [tool_condition_labels.get(val, "Unknown") for val in pred_tool_condition]
    st.write("Tool Condition Predictions:")
    st.write(pred_tool_condition)

# Button to predict all at once
if st.button("Predict All"):
    predictions = model.predict(df_test).round().astype(int)
    
    results = pd.DataFrame({
        "Passed_Visual_Inspection": [passed_visual_labels.get(val, "Unknown") for val in predictions[:, 0]],
        "Machining_Finalized": [machining_finalized_labels.get(val, "Unknown") for val in predictions[:, 1]],
        "Tool_Condition": [tool_condition_labels.get(val, "Unknown") for val in predictions[:, 2]]
    })
    
    st.write("All Predictions:")
    st.write(results.head())

    # Save results
    results_csv = results.to_csv(index=False)
    st.download_button("Download Predictions", results_csv, "prediction_results.csv", "text/csv")
