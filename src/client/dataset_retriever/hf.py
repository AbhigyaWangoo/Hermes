from huggingface_hub import HfApi, snapshot_download
from typing import List, Any
import os
from . import base
from include.utils import DATA_DIR

DEFAULT_TIMEOUT = 5
DATASET_DIR = os.path.join(DATA_DIR, "datasets")

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)


class HuggingFaceClient(base.AbstractDatasetClient):
    """A client to interact with hugging face datasets"""

    def __init__(self):
        self.client = HfApi()

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

    def download_dataset(self, dataset_id: str, local_filepath: str):
        """
        Download a dataset from the Hugging Face Hub and save it to a local filepath.
        """

        try:
            if not os.path.exists(local_filepath):
                os.mkdir(local_filepath)

            snapshot_download(
                repo_id=dataset_id, repo_type="dataset", local_dir=local_filepath
            )

            print(f"Dataset '{dataset_id}' downloaded to '{local_filepath}'")
            return True
        except Exception as e:
            print(f"Error downloading dataset '{dataset_id}': {e}")
            return False
