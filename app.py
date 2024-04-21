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
    # vector_store = MongoDBAtlasVectorSearch(mongo.client, mongo.database_name, mongo.collection_name, index_name="vector_index", embedding_key="dataset_embedding", text_key="dataset_summary", id_key="_id", metadata_key="links")
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(similarity_top_k=5)
    # query_embedding = gpt_client.generate_embeddings(sentence=query)
    # print("query_embedding", type(query_embedding), query_embedding)
    
    response = query_engine.query(dataset_description)
    print(type(response))
    print(str(response))


st.set_page_config(page_title="Dataset Finder", page_icon="📊🔍")
st.title("📊🔍 Dataset Finder with MongoDB")

st.write("We know how hard it is to find the right dataset for your task, let us help you!")

uploaded_file = st.file_uploader("Upload a CSV with a few examples of the data you're looking for", type=("csv"))

dataset_description = st.text_input(
    "Don't have a CSV? No problem, describe the kind of dataset you would like",
    placeholder="A pets dataset with attributes like name, breed, age, etc",
    disabled=not not uploaded_file,
)

submit_button = st.button(
    "Submit",
    disabled= not uploaded_file and not dataset_description
)

if submit_button: 
    # if uploaded_file:
    #     examples = pd.read_csv(uploaded_file)
    # else: 
    #     examples = pd.DataFrame([dataset_description])

    if uploaded_file:
        print("user uploaded csv file")
        # User uploaded a file, create embeddings using MongoDB Atlas Vector Search 

        # Embed each example (row) in the CSV
        embeddings = [] 
        for example in uploaded_file:
            embeddings.append(gpt_client.generate_embeddings(sentence=example))

        print("embeddings done")
        for embedding in embeddings:
            print("embedding:", embedding)
            mongo.perform_vector_search(embedding)
    else: 
        # User submitted a NL query -- aka a description of the dataset they're searching for 
        response = retrieve_datasets_NLQ(dataset_description)


# add a try again button 


# FOR LATER:
# slider for how many datasets they want back 
# require openAI key input 


