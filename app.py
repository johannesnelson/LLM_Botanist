import streamlit as st
import pandas as pd
import data_processing as dp
import prompt_chat_config as pcc
from dotenv import load_dotenv
import os
import openai

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
        input_species = st.text_input("Species (scientific name)")
        input_country = st.text_input("Country")
        submit_button = st.form_submit_button(label='Process Single Entry')


    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    elif submit_button:
        if input_species and input_country:
            df = pd.DataFrame({'species': [input_species], 'country': [input_country]})
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
            csv = processed_df.to_csv(index=False)  # Set index=False to exclude row indices from the CSV
            st.download_button(
                label="Download data as CSV",  # Text on the download button
                data=csv,  # Data to be downloaded
                file_name='processed_data.csv',  # Name of the file (users will download the file with this name)
                mime='text/csv',  # Mime type (this indicates it is a CSV file)
    )
    else:
        st.warning("Please upload a CSV file or enter the species and country details.")
    st.markdown("""
    ### Overview            
    This tool uses GPT 3.5 Turbo from OpenAI to help classify plant species in certain countries as either native or alien,
    based solely on context from Wikipedia to avoid hallucinations. To accomplish this, it first queries Wikipedia for information
    about the plant. It filters theresults of this query with keywords to help cut down on the context sent to the LLM.
                
    Then, using a prompt schema designed with LangChain, the LLM attempts to determine the native range and the alien range of the plant. The model is then
    instructed to explicitly state its reasoning using these delineated ranges before making a classification 
    decision. If no Wikipedia context is found, no decision is made. The output is a dataframe with a classifcation decision and all the context
    extracted and used by the LLM when arriving at it.
                
    You can either run it with a single species, country combination, or by uploading a CSV with a 'species' column and a 'country' column.
    
    ### Note 
    For now, my own OpenAI API key is used, and since I don't anticipate a lot of traffic, that is fine. If you
    happened to find this and would like to use it for a larger dataset, please reach out first and I can help you 
    set up your own API key.
            """)

if __name__ == "__main__":
    main()