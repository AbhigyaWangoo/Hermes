import os
import pandas as pd
from typing import List, Dict, Any
import openai
from src.client.mongo import MongoDBUploader
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE=100

class Chunker:
    """ A class to chunk up a dataset and place each chunk into mongodb atlas """
    def __init__(self, chunk_size: int=CHUNK_SIZE):
        self.chunk_size = chunk_size

        api_key=os.environ.get("OPENAI_API_KEY", None)
        mongodb_url=os.environ.get("MONGODB_URL", None)
        mongodb_db=os.environ.get("MONGODB_DB", None)
        mongodb_collection=os.environ.get("MONGODB_COLLECTION", None)

        self.mongo_client = MongoDBUploader(mongodb_url, mongodb_db, mongodb_collection)
        self.openai_client = openai.OpenAI(api_key=api_key)

    def read_csv(self, file_path: str):
        """
        Read the CSV file and return a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print("File not found.")
            return None

    def split_into_chunks(self, data: pd.DataFrame):
        """
        Split the DataFrame into chunks of size self.chunk_size.
        """
        num_chunks = len(data) // self.chunk_size
        return [data[i*self.chunk_size:(i+1)*self.chunk_size] for i in range(num_chunks)]

    def embed(
        self, content: str, embedding_model: str = "text-embedding-3-small"
    ) -> List[str]:
        """
        Vectorize a sentence using OpenAI's GPT-3 model.

        Args:
            content (str): The input sentence to vectorize.
            embedding_model (str): The embedding model to use.

        Returns:
            list: The vector representation of the sentence.
        """

        response = self.openai_client.embeddings.create(
            model=embedding_model, input=content  # Choose the appropriate engine
        )

        # Extract the vector representation from the response
        return response.data[0].embedding

    def process_file(self, file_path: str):
        """
        Read the CSV file, split it into chunks, and encode each chunk.
        """
        data = self.read_csv(file_path)

        if data is not None:
            chunks = self.split_into_chunks(data)
            embedded_chunks = {}
            for idx, chunk in enumerate(chunks):
                embedded_chunks[f"{file_path}_{idx}"] = self.embed(chunk)
            return embedded_chunks

    def upload_chunks_to_mongo(self, chunks: Dict[str, List[Any]]):
        """ Upload a set of chunk embeddings into mongodb with the filename+chunksize """
        self.mongo_client.upload_embeddings(chunks)
