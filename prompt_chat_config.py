
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
import pandas as pd
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from json.decoder import JSONDecodeError

def prepare_LLM(llm_model = "gpt-3.5-turbo"):
    llm_model = llm_model
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    return chat



def prepare_template_string():
    
    template_string = """You are acting as an ecology/botany assistant. Your \
    job is to help designate a plant species as either 'native', 'alien', or \
    'unknown' given the species name, the country it is being planted in, and \
    contextual information. This provided context is the only information you \
    may use when making a decision. If the context is unavailable or irrelevant,\
    your decision must be 'unknown'. You must never decide a plant is native or \
    alien unless you have been provided conclusive context, even if you \
    think you know the answer.You must also be very careful to only consider the\
    plant within the country under review. The context may contain information \
    about its status in other regions, but you must stay focused when making a \
    decision to the country that is provided in the 'country' part of the prompt \
    below. Begin with identifying the native range, then identify the alien range, \
    if possible. Then, state your reasoning based on these ranges, organizing your \
    response to follow the format_instructions in the prompt. Do not forget the \
    backticks to delineate the json obect."
    
    The country, species, and context will be delimited by the triple backticks below. \
    Format instructions will be provided.
    
    ``` 
    Species: {species}, Country: {country}, Context: {context}
    ```
    {format_instructions}
    """
    return template_string

def prepare_chat_schemas():
        
    species_schema = ResponseSchema(name="species",
                                     description="Simply state the species under review.")
    country_schema = ResponseSchema(name="country",
                                     description="Simply state the country where the given species "
                                     "is being reviewed for status.")
    
    
    native_range_schema = ResponseSchema(name="native_range",
                                     description= "From the context provided, ascertain the native range of the plant species, and write it here "
                                        "as a single string. If there is no clear native range given in the context, state: unable to ascertain native range "
                                         "from provided context. This should be a single string.")
    
    alien_range_schema = ResponseSchema(name="alien_range",
                                     description= "From the context provided, ascertain the alien range of the plant species, and write it here "
                                        "as a single string. There may not be mention of an alien or introduced range, in which case you should simply "
                                       "state: unable to ascertain alien range. This should be a single string.")
    
    reasoning_schema = ResponseSchema(name="reasoning",
                                     description= "Based upon the native range and alien range that you extracted, explain the reasoning that leads "
                                      "to your decision about the specific country under review. Remember that the country name may not be explicitly mentioned "
                                     "but that it might exist within the broader continental or regional boundaries that are provided. Explain your reasoning "
                                     "clearly about this. If you first identify it as native because of the extracted range, then stick to that decision. "
                                     "This should be a single string.")
    
    decision_schema = ResponseSchema(name="decision",
                                     description=" Acceptable decisions: native, alien, or unknown. "
                                    "The decision must be made entirely from the given context and must be specific to the species and "
                                     "country provided in this prompt. If there is no context provided, the decision "
                                    "must be 'unknown'. If and only if there is relevant context are you allowed to make "
                                    "a decision of 'native' or 'alien'. Use your extracted native range as the first, primary piece of evidence. "
                                    "Even if the country name is not explicitly mentioned in the native range description, you must decide if the country "
                                    "falls within the described range. For example, some ranges may be described at larger scales like regions and continents, "
                                    "so even if the country is not mentioned, you have to decide if it falls within these broader boundaries. If the country "
                                    "falls within this native range, you must decide that it is native. If it is not within this native range, it is likely alien "
                                    "which you can verify if there is any alien range context that you were able to extract.")
    
    
    
    information_source_schema = ResponseSchema(name="information_source",
                                          description="If there was no context provided, this must state 'no context provided' "
                                               "If there was relevant context provided, you must provide here the exact, verbatim sentences "
                                              "from the provided context that you used to make your "
                                              "decision. If there was no context, state that there was "
                                              "no context provided. In the cases where the context was irrelevant "
                                              "or insufficient, state so explicitly here. This should be a single string and it should contain no "
                                               "more than three sentences.")
    
    response_schemas = [species_schema, 
                        country_schema,
                        native_range_schema,
                        alien_range_schema,
                        reasoning_schema,
                        decision_schema,
                        information_source_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return response_schemas, output_parser, format_instructions
