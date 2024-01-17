import streamlit as st
import pandas as pd
import data_processing as dp
import prompt_chat_config as pcc
from dotenv import load_dotenv
import os


# Load API key from .env
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")



def main():
    st.title("LLM-Powered Native/Alien Species Classifier")
    df = None
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    with st.form("single_species_form"):
        st.write("Or enter details manually:")
        input_species = st.text_input("Species")
        input_country = st.text_input("Country")
        submit_button = st.form_submit_button(label='Process Single Entry')

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    elif submit_button:
        if input_species and input_country:
            df = pd.DataFrame({'species': [input_species], 'country_name': [input_country]})
        else:
            st.warning("Please enter both species and country.")

    if df is not None:
        chat = pcc.prepare_LLM()
        template_string = pcc.prepare_template_string()
        response_schemas, output_parser, format_instructions = pcc.prepare_chat_schemas()
        
        # Process data
        if st.button('Process Data') | submit_button:

            progress_bar = st.progress(0)  # Initialize the progress bar
            
            def update_progress(progress):
                progress_bar.progress(progress)

            st.text("Processing data...")
            processed_df = dp.process_species_data(df, chat=chat, output_parser=output_parser, template_string=template_string, format_instructions=format_instructions, progress_callback=update_progress)

            # Display processed data
            st.write(processed_df)
    else:
        st.warning("Please upload a CSV file or enter the species and country details.")


if __name__ == "__main__":
    main()