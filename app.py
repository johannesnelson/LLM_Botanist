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

    # File uploader widget
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Prepare chat and templates
        chat = pcc.prepare_LLM()
        template_string = pcc.prepare_template_string()
        response_schemas, output_parser, format_instructions = pcc.prepare_chat_schemas()

        # Process data
        if st.button('Process Data'):
            progress_bar = st.progress(0)  # Initialize the progress bar
            
            def update_progress(progress):
                progress_bar.progress(progress)

            st.text("Processing data...")
            processed_df = dp.process_species_data(df, chat=chat, output_parser=output_parser, template_string=template_string, format_instructions=format_instructions, progress_callback=update_progress)

            # Display processed data
            st.write(processed_df)

if __name__ == "__main__":
    main()