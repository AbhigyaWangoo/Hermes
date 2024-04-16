from abc import ABC, abstractmethod
from typing import List, Any

class AbstractDatasetClient(ABC):
    """
    An abstract class for dataset clients
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
