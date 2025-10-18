import streamlit as st
import pandas as pd
import plotly.express as px
from .model_ops import list_org_models, detect_bias, load_model

def upload_csv():
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("### Preview", df.head())
        return df

def select_model(key_prefix):
    st.write("### Select Model")
    try:
        models = list_org_models()
        if not models:
            st.warning("No public models found.")
            return None

        model_names = [m.modelId for m in models]
        return st.selectbox("Choose a model:", model_names, key=key_prefix+"model_select_box")

    except Exception as e:
        st.warning(f"Error: {e}")
        return None
    
def test_model(model_id):
    model, tokenizer = load_model(model_id)
    st.write("### Try custom text")
    user_text = st.text_area("Enter text to test:", "")

    if user_text and st.button("Check bias on this text"):
        if model is None:
            st.error("No model selected!")
        else:
            # Create a temporary placeholder for the "Analyzing..." message
            analyzing_placeholder = st.empty()
            analyzing_placeholder.info("Analyzing custom text…")

            # Run bias detection
            label, certainty = detect_bias(user_text, model, tokenizer)

            # Clear the placeholder message
            analyzing_placeholder.empty()

            # Show the result
            if label == 1:
                st.error("⚠️ The model detected bias in the text.")
            else:
                st.success("✅ No bias detected in the text.")

def run_analysis(df, model_id):
    model, tokenizer = load_model(model_id)

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    col = st.selectbox("Select text column:", text_columns, key="column_select_box")

    if st.button("Run Analysis"):
        # Placeholder for "Running…" info
        analyzing_placeholder = st.empty()
        analyzing_placeholder.info("Running bias detection… this may take a moment.")

        results = []
        progress = st.progress(0)

        # Detect bias
        for i, text in enumerate(df[col].fillna("")):
            results.append(detect_bias(text, model, tokenizer)[0])
            progress.progress((i + 1) / len(df))

        df["bias_detected"] = results

        # Clear analyzing info
        analyzing_placeholder.empty()
        st.success("Analysis complete!")

        # Split into biased / non-biased
        df_biased = df[df["bias_detected"] == 1].copy()
        df_non_biased = df[df["bias_detected"] == 0].copy()

        # --------------------------
        # Display tables in two columns
        # --------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bias not detected ✅")
            if not df_non_biased.empty:
                st.dataframe(
                    df_non_biased.style.set_properties(
                        **{"background-color": "#00CC96"}  
                    ),
                    height=400,
                    use_container_width=True
                )
            else:
                st.info("No non-biased rows found.")

        with col2:
            st.subheader("Bias detected ⚠️")
            if not df_biased.empty:
                st.dataframe(
                    df_biased.style.set_properties(
                        **{"background-color": "#EF553B"}  
                    ),
                    height=400,
                    use_container_width=True
                )
            else:
                st.info("No biased rows found.")

        # --------------------------
        # Pie chart and counts in two columns
        # --------------------------
        pie_col, count_col = st.columns([2, 1])  

        # Pie chart
        counts = df["bias_detected"].value_counts().rename(index={0: "Non-Biased", 1: "Biased"})
        pie_data = counts.reset_index()
        pie_data.columns = ["Label", "Count"]

        with pie_col:
            fig = px.pie(
                pie_data,
                names="Label",
                values="Count",
                title="Distribution",
                color="Label",
                color_discrete_map={"Biased": "#EF553B", "Non-Biased": "#00CC96"},
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        # Row counts summary
        with count_col:
            st.subheader("Summary")
            st.write(f"**Total rows:** {len(df)}")
            st.write(f"**Non-biased rows:** {len(df_non_biased)}")
            st.write(f"**Biased rows:** {len(df_biased)}")
