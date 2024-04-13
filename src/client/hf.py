from huggingface_hub import HfApi, hf_hub_download
import requests
import pandas as pd

DEFAULT_TIMEOUT=5

class HuggingFaceClient:
    """A client to interact with hugging face datasets"""

    def __init__(self):
        self.client=HfApi()

    def list_text_datasets(self) -> set:
        """
        List all Hugging Face text-based datasets.
        """
        try:
            datasets = self.client.list_models()

            txt_sets = set()
            for dataset in datasets:
                if "text" in dataset.tags:
                    txt_sets.add(dataset.id)

            return txt_sets
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            return None

    def download_dataset(self, repo_id: str, local_filepath: str):
        """
        Download a dataset from the Hugging Face Hub and save it to a local filepath.
        """

        try:
            dataset = pd.read_csv(
                hf_hub_download(repo_id=repo_id, filename=local_filepath, repo_type="dataset")
            )

            dataset.to_csv(local_filepath)

            print(f"Dataset '{repo_id}' downloaded to '{local_filepath}'")
            return True
        except Exception as e:
            print(f"Error downloading dataset '{repo_id}': {e}")
            return False
