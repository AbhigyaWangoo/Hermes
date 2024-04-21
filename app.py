import streamlit as st
import pandas as pd
from dotenv import load_dotenv 
import os
# from client.gpt import GPTClient
from src.client.gpt import GPTClient
from src.client.mongo import MongoDBUploader
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_response
from IPython.display import Markdown

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

mongo = MongoDBUploader(MONGODB_URL, MONGODB_DB, MONGODB_COLLECTION)
gpt_client = GPTClient()

# Retrieves datasets using LlamaIndex 
def retrieve_datasets_NLQ(dataset_description): 
    vector_store = MongoDBAtlasVectorSearch(mongo.client)
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(similarity_top_k=5)
    
    response = query_engine.query(dataset_description)
    print(type(response))
    print(str(response))

def calc_dataset_freq(results):
    dataset_freq = {}
    for result in results: 
        for dataset in result:
            print("dataset + metadata", dataset)
            dataset_name = "".join(dataset['_id'].split("_")[:-1])
            print(dataset_name)
            if dataset_name in dataset_freq:
                dataset_freq[dataset_name]["freq"] += 1
            else:
                dataset_freq[dataset_name] = {"freq": 1, "name": dataset_name, "links": dataset["links"], "dataset_summary": dataset["dataset_summary"]}
    return dataset_freq

st.set_page_config(page_title="Dataset Finder", page_icon="📊🔍")
st.title("📊🔍 Dataset Finder with MongoDB")

st.write("We know how hard it is to find the right dataset for your task, let us help you!")

uploaded_file = st.file_uploader("Upload a CSV with a few examples of the data you're looking for", type=("csv"))

dataset_description = st.text_input(
    "Don't have a CSV? No problem, describe the kind of dataset you would like",
    placeholder="A pets dataset with attributes like name, breed, age, etc",
)

submit_button = st.button(
    "Submit",
)

if submit_button: 
    examples = []
    if uploaded_file:
        for example in uploaded_file:
            examples.append(example)
    if dataset_description:
        examples.append(dataset_description)
    

    # Create embeddings using MongoDB Atlas Vector Search 
    # Embed each example (row) in the CSV
    embeddings = [] 
    for example in examples:
        embeddings.append(gpt_client.generate_embeddings(sentence=example))

    all_results = []
    for embedding in embeddings:
        all_results.append(mongo.perform_vector_search(embedding))
    
    print("all_results", all_results)
    dataset_freq = calc_dataset_freq(all_results)
    print("dataset_freq", dataset_freq)

    # for dataset in 
    sorted_frequency_list = sorted(dataset_freq.items(), key=lambda x: x[1]['freq'], reverse=True)

    print("sorted freq list", sorted_frequency_list)


# once we get a response from 
# st.write_stream()
# st.write()

# add a try again button 


# FOR LATER:
# slider for how many datasets they want back 
# require openAI key input 


