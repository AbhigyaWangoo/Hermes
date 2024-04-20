from src.client.dataset_retriever.hf import HuggingFaceClient, DATASET_DIR
import os
from src.process import process_single_dataset

if __name__ == "__main__":

    dcl = HuggingFaceClient()
    dtp = dcl.list_datasets("text-classification")

    for dataset in dtp:
        print(dataset.id)
        name = dataset.id
        dataset_dst = os.path.join(DATASET_DIR, name.split("/")[-1])
        if not os.path.exists(dataset_dst):
            process_single_dataset(name, dataset_dst, dcl)
        break
