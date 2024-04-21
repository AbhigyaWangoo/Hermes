import os
import chardet
import tqdm
import random
import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any
import numpy as np
from src.client.mongo import MongoDBUploader
from src.client.gpt import GPTClient
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 100
DEFAULT_NUM_CHUNKS=300

class Chunker:
    """A class to chunk up a dataset and place each chunk into mongodb atlas"""

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size

        mongodb_url = os.environ.get("MONGODB_URL", None)
        mongodb_db = os.environ.get("MONGODB_DB", None)
        mongodb_collection = os.environ.get("MONGODB_COLLECTION", None)

        self.mongo_client = MongoDBUploader(mongodb_url, mongodb_db, mongodb_collection)
        self.openai_client = GPTClient()

    def split_into_chunks(
        self,
        data: pd.DataFrame,
        n_chunks: int = DEFAULT_NUM_CHUNKS,
        split_by_line: bool = True,
    ) -> List[str]:
        """
        Split the DataFrame into chunks of size self.chunk_size.
        
        TODO splitting by string chunks seems to be broken right now. split by line for default.

        TODO splitting by string chunks seems to be broken right now. split by line for default.

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
                chunk = data.iloc[i : i + self.chunk_size].to_string(index=False)
                chunks.append(chunk)
        else:
            random_indices = random.sample(range(len(data)), n_chunks)
            chunks = [
                data.iloc[index].to_string(index=False) for index in random_indices
            ]

        rand_chunks = np.random.choice(chunks, n_chunks)
        return rand_chunks

    def read_into_df(self, file_path: str) -> pd.DataFrame:
        """A helper function to read any file type into a df"""

        file_formats = {
            ".csv": "csv",
            ".json": "json",
            ".txt": "txt",
            ".parquet": "parquet",
        }

        file_formats = {'.csv': 'csv', '.json': 'json', '.txt': 'txt', '.parquet': 'parquet'}

        file_type = None

        _, ext = os.path.splitext(file_path)
        if ext in file_formats:
            file_type = file_formats[ext]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        encoding=None
        with open(file_path, 'rb') as f:
            rawdata = f.read(1024)
            result = chardet.detect(rawdata)
            encoding = result['encoding']

        print(encoding)

        try:
            # dataset_dict = load_dataset(file_type, data_files=file_path)
            # df = pd.concat([dataset.to_pandas(encoding=encoding) for dataset in tqdm.tqdm(dataset_dict.values(), desc="Reading dataset values and concatenating into a dataframe")], ignore_index=True)
            if file_type == 'csv':
                df = pd.read_csv(file_path, encoding=encoding)
            elif file_type == 'json':
                df = pd.read_json(file_path, encoding=encoding)
            elif file_type == 'txt':
                df = pd.read_csv(file_path, delimiter='\t', encoding=encoding)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path, engine='pyarrow')
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except ValueError as ve:
            print(f"Error when loading dataset file {file_path}. Erroring out: {ve}")
            raise ValueError from ve

        return df

    def process_files(
        self, files: List[str], dataset_name: str
    ) -> Dict[str, List[List[float]]]:
        """
        This wrapper coalesces multiple files into one single dataframe to chunk.
        """
        rv_df = pd.DataFrame()

        for fstr in files:
            try:
                df = self.read_into_df(fstr)
                rv_df = rv_df.append(df, ignore_index=True)
            except ValueError as e:
                print(f"Couldn't load {fstr}, continuing without it. Error: {e} ")

        if len(rv_df) == 0:
            raise ValueError(f"Couldn't load an entire dataset {dataset_name}")

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
            for idx, chunk in tqdm.tqdm(
                enumerate(chunks),
                desc=f"Embedding {len(chunks)} chunks for dataset {dataset_name}",
            ):
                embeddings = self.openai_client.generate_embeddings(chunk)
                embedded_chunks[f"{dataset_name}_{idx}"] = embeddings
            return embedded_chunks

    def upload_chunks_to_mongo(
        self,
        chunks: Dict[str, List[List[float]]],
        links: List[str] = None,
        dataset_summary: str = None,
    ):
        """Upload a set of chunk embeddings into mongodb with the filename+chunksize"""
        self.mongo_client.upload_embeddings(
            chunks, links=links, dataset_summary=dataset_summary
        )
