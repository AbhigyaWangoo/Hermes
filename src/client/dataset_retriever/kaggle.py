import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import List, Any
from . import base

class KaggleDatasetClient(base.AbstractDatasetClient):
    """
    An abstract class for dataset clients
    """
    
    def __init__(self) -> None:
        self.api = KaggleApi()
        self.api.authenticate()

    def get_data_file_from_dir(self, directory: str) -> List[str]:
        """
        An abstract method that given a directory, returns
        the path of all files within that directory.
        """

    def download_dataset(self, dataset_id: str, local_filepath: str):
        """
        An abstract method to download a dataset provided with some
        identifier to a local filepath
        """

    def list_datasets(self, dataset_filter: str) -> List[Any]:
        """
        Lists Kaggle datasets matching a filter.
        """
        try:
            rv=[]
            competitions = self.api.competitions_list()

            for competition in competitions:
                print(competition)
                files=self.api.competitions_data_list_files(competition)
                print(files)
                rv += files

            return rv
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            return None

    def generate_dataset_summary(self, dataset_id: str) -> str:
        """
        An abstract method to generate a dataset summary.
        """

    def generate_dataset_link(self, dataset_id: str) -> str:
        """
        An abstract method to get the link to the dataset.
        """
