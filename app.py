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

demo_data = {
    'species': [
        'Acacia binervia', 'Acacia brachybotrya', 'Adenanthera pavonina',
        'Albizia lebbeck', 'Andira anthelmia',
        'Andira fraxinifolia', 'Cedrela odorata'
    ],
    'country': [
        'Australia', 'Australia', 'Democratic Republic of Congo',
        'Madagascar', 'Brazil',
        'Brazil', 'Mexico'
    ]
}
demo_data = pd.DataFrame(demo_data)


def main():
    st.title("LLM-Powered Native/Alien Species Classifier")

    df = None
    # File uploader widget
    demo_button = st.button('Run Demo')
    if demo_button:
        df = pd.DataFrame(demo_data)
        st.markdown("""
        ## This is the demo dataset that is being processed:
                    """)
        st.write(demo_data)
    uploaded_file = st.file_uploader("Upload a CSV file with a 'species' and a 'country' column", type="csv")
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
        if st.button('Process Data') | submit_button | demo_button:

            progress_bar = st.progress(0)  # Initialize the progress bar
            
            def update_progress(progress):
                progress_bar.progress(progress)

            st.text("Gathering data from Wikipedia. This step can take some time with larger datasets!")
            processed_df = dp.process_species_data(df, chat=chat, output_parser=output_parser, template_string=template_string, format_instructions=format_instructions, progress_callback=update_progress)

            # Display processed data
            st.markdown("""
                        ## Results
                        """)
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
    based solely on context from Wikipedia. To accomplish this, it first queries Wikipedia for information
    about the plant. It filters the results of this query with keywords to help cut down on the context sent to the LLM.
                
    Then, using a prompt schema designed with LangChain, the LLM attempts to determine the native range and the alien range of the plant. The model is then
    instructed to explicitly state its reasoning using these delineated ranges before making a classification 
    decision. If no Wikipedia context is found, no decision is made. The output is a dataframe with a classifcation decision and all the context
    extracted and used by the LLM when arriving at it.
                
    You can either run it with a single species-country combination entered manually or by uploading a CSV with a 'species' column and a 'country' column.
    ### Interpreting Results
    The results will show the country, species, whatever wikipedia context was extracted, the explicit alien and native ranges, the reasoning
    steps that the LLM used, and the cited part of the context that was used to make the decision. However, the non-deterministic nature of LLMs
    will sometimes introduce unwanted results. This is especially true if the species names are not spelled correctly, but can happen in cases where
    no context was provided at all. 
                
    ### Note about API usage
    For now, my own OpenAI API key is used, and since I don't anticipate a lot of traffic, that is fine. If you
    happened to find this and would like to use it for a larger dataset, please reach out first and I can help you 
    set up your own API key.
            """)

if __name__ == "__main__":
    main()