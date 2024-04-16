from huggingface_hub import HfApi, snapshot_download
from typing import List, Any
import pandas as pd

DEFAULT_TIMEOUT=5

class HuggingFaceClient:
    """A client to interact with hugging face datasets"""

    def __init__(self):
        self.client=HfApi()

    def list_datasets(self, dataset_filter: str) -> List[Any]:
        """
        List all Hugging Face text-based datasets. Wrapper around 
        HFAPI for future caching.
        
        returns: List of DatasetInfo objects.
        """
        try:
            datasets = self.client.list_datasets(filter=dataset_filter)

            return datasets
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            return None

    def download_dataset(self, repo_id: str, local_filepath: str):
        """
        Download a dataset from the Hugging Face Hub and save it to a local filepath.
        """

        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_filepath)

            print(f"Dataset '{repo_id}' downloaded to '{local_filepath}'")
            return True
        except Exception as e:
            print(f"Error downloading dataset '{repo_id}': {e}")
            return False
