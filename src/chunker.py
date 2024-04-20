import os
import random
import pandas as pd
from typing import List, Dict, Any
import numpy as np
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

    def split_into_chunks(self, data: pd.DataFrame, n_chunks: int = 3, split_by_line: bool=True) -> List[str]:
        """
        Split the DataFrame into chunks of size self.chunk_size.

        data: df with data.
        n_chunks: The number of chunks to return. If < 0, returns the entire chunked dataset.
                  Else, this will sample n_chunks chunks from the entire dataframe
        split_by_line: If true, this will split the data by each row, and not by raw data.
        """
        if n_chunks < 0:
            n_chunks = len(data) // self.chunk_size

        chunks = []
        if not split_by_line:
            for i in range(0, len(data), self.chunk_size):
                chunk = data.iloc[i : i + self.chunk_size]
                chunks.append(chunk)
        else:
            random_indices = random.sample(range(len(data)), n_chunks)
            chunks = [data.iloc[index].to_string(index=False) for index in random_indices]
            print(len(chunks))

        rand_chunks = np.random.choice(chunks, n_chunks)
        return rand_chunks

    def process_files(
        self, files: List[str], dataset_name: str
    ) -> Dict[str, List[List[float]]]:
        """
        This wrapper coalesces multiple files into one single dataframe to chunk.
        """
        rv_df: pd.DataFrame
        with open(files[0], "r", encoding="utf8") as fp:
            rv_df = pd.read_csv(fp)

        for i in range(1, len(files), 1):
            fstr = files[i]
            with open(fstr, "r", encoding="utf8") as fp:
                df = pd.read_csv(fp)

                rv_df = rv_df.append(df, ignore_index=True)

        return self.process_dataset(rv_df, dataset_name)

    def process_dataset(
        self, data: pd.DataFrame, dataset_name: str
    ) -> Dict[str, List[List[float]]]:
        """
        Read the CSV file, split it into chunks, and encode each chunk.

        returns: A dict of the following type {"filename_index" : [[chunk embedding]]}
        """
        if data is not None:
            chunks = self.split_into_chunks(data)
            embedded_chunks = {}
            for idx, chunk in enumerate(chunks):
                embedded_chunks[f"{dataset_name}_{idx}"] = (
                    self.openai_client.generate_embeddings(chunk)
                )
            return embedded_chunks

    def upload_chunks_to_mongo(self, chunks: Dict[str, List[Any]]):
        """Upload a set of chunk embeddings into mongodb with the filename+chunksize"""
        self.mongo_client.upload_embeddings(chunks)
