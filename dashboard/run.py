import streamlit as st
from utils.components import run_analysis, select_model, upload_csv, test_model

st.set_page_config(page_title="Bias Detection Dashboard", layout="wide")
st.title("Bias Detection Dashboard")


tab1, tab2 = st.tabs(["Test a Model", "Detect Bias in Dataset"])


with tab1:
    st.header("Try the model with custom input")
    model_id = select_model(key_prefix="for_trying_model_out")
    if model_id:
        test_model(model_id)

with tab2:
    st.header("Detect bias in a dataset")
    uploaded_df = upload_csv()
    if uploaded_df is not None:
        model_id2 = select_model(key_prefix="for_dataset_analysis")
        if model_id2:
            run_analysis(uploaded_df, model_id2)
