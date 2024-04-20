from src.client.dataset_retriever.hf import HuggingFaceClient, DATASET_DIR
import os
from src.process import process_single_dataset
from include.utils import clear_directory

if __name__ == "__main__":

    dcl = HuggingFaceClient()
    dtp = dcl.list_datasets("text-classification")

    count=10
    for dataset in dtp:
        name = dataset.id
        dataset_dst = os.path.join(DATASET_DIR, name.split("/")[-1])

        if os.path.exists(dataset_dst):
            continue

        process_single_dataset(name, dataset_dst, dcl)
        clear_directory(dataset_dst) # Clearing up disk space, as we have the data in mongodb atlas now.

        count-=1
        if count == 0:
            break
