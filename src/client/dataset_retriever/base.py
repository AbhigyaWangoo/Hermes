from abc import ABC, abstractmethod
from typing import List, Any
from src.client.gpt import GPTClient

DATASET_SUMMMARY_PROMPT = """
You are provided with the following text that contains information regarding a dataset. Your job is to provide me with a comprehensive summary
of the entire text in a format that is cohesive and easy to understand. Please ensure that the summary includes relevant details and examples that 
support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length 
and complexity of the original text, providing a clear and accurate overview without omitting any important information.
"""


class AbstractDatasetClient(ABC):
    """
    An abstract class for dataset clients
    """

    def __init__(self) -> None:
        self.gpt_client = GPTClient()

    @abstractmethod
    def get_data_file_from_dir(self, directory: str) -> List[str]:
        """
        An abstract method that given a directory, returns
        the path of all files within that directory.
        """

    @abstractmethod
    def download_dataset(self, dataset_id: str, local_filepath: str):
        """
        An abstract method to download a dataset provided with some
        identifier to a local filepath
        """

    @abstractmethod
    def list_datasets(self, dataset_filter: str) -> List[Any]:
        """
        An abstract method to list a set of datasets for
        processing.
        """

    @abstractmethod
    def generate_dataset_summary(self, dataset_id: str) -> str:
        """
        An abstract method to generate a dataset summary.
        """

    @abstractmethod
    def generate_dataset_link(self, dataset_id: str) -> str:
        """
        An abstract method to get the link to the dataset.
        """
