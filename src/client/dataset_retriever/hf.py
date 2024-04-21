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

    def __init__(self) -> None:
        super().__init__()
        self.client = HfApi()

    def list_datasets(self, dataset_filter: str) -> List[Any]:
        """
        List all Hugging Face text-based datasets. Wrapper around
        HFAPI for future caching.

        returns: List of DatasetInfo objects.
        """
        try:
            datasets_found = self.client.list_datasets(filter=dataset_filter)

            return datasets_found
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

    def generate_dataset_link(self, dataset_id: str) -> str:
        """
        Gather the link to a huggingface dataset.

        dataset_id: the name of the dataset in hf
        """
        base_url = "https://huggingface.co/datasets/"
        dataset_url = os.path.join(base_url, dataset_id)

        return dataset_url

    def generate_dataset_summary(self, dataset_id: str) -> str:
        """
        Get the summary of the huggingface dataset. Reads the
        readme file and generates a summary based on that.

        dataset_id: the root dir of the dataset
        """

        for root, _, files in os.walk(dataset_id):
            for file in files:
                if file.lower() == "readme.md":
                    with open(os.path.join(root, file), "r", encoding="utf8") as fp:
                        documentation = fp.read().strip()

                        summary = self.gpt_client.query(
                            prompt=documentation,
                            sys_prompt=base.DATASET_SUMMMARY_PROMPT,
                        )

                        return summary

    def get_data_file_from_dir(self, directory: str) -> List[str]:
        """
        Given a directory, this returns the path of all files within that directory
        (as well as nested files) that have 'train' or 'test' in their name. The
        output is a list of these paths.

        TODO look at this, see if u can make it better: https://huggingface.co/docs/datasets/loading
        """

        data_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if "train" in file or "test" in file or "valid" in file:
                    data_files.append(os.path.join(root, file))

        return data_files
