import os
import pandas as pd
from typing import List, Dict, Any
import openai
from src.client.mongo import MongoDBUploader
from src.client.gpt import GPTClient
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 100

class Chunker:
    """A class to chunk up a dataset and place each chunk into mongodb atlas"""

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size

        mongodb_url = os.environ.get("MONGODB_URL", None)
        mongodb_db = os.environ.get("MONGODB_DB", None)
        mongodb_collection = os.environ.get("MONGODB_COLLECTION", None)

        self.mongo_client = MongoDBUploader(mongodb_url, mongodb_db, mongodb_collection)
        self.openai_client = GPTClient()

    def split_into_chunks(self, data: pd.DataFrame):
        """
        Split the DataFrame into chunks of size self.chunk_size.
        """
        num_chunks = len(data) // self.chunk_size
        return [
            data[i * self.chunk_size : (i + 1) * self.chunk_size]
            for i in range(num_chunks)
        ]

    def process_file(self, file_path: str) -> Dict[str, List[List[float]]]:
        """
        Read the CSV file, split it into chunks, and encode each chunk.

        returns: A dict of the following type {"filename_index" : [[chunk embedding]]}
        """
        data = pd.read_csv(file_path)

        if data is not None:
            chunks = self.split_into_chunks(data)
            embedded_chunks = {}
            for idx, chunk in enumerate(chunks):
                embedded_chunks[f"{file_path}_{idx}"] = (
                    self.openai_client.generate_embeddings(chunk)
                )
            return embedded_chunks

    def upload_chunks_to_mongo(self, chunks: Dict[str, List[Any]]):
        """Upload a set of chunk embeddings into mongodb with the filename+chunksize"""
        self.mongo_client.upload_embeddings(chunks)
