import os
from src.client.dataset_retriever.base import AbstractDatasetClient
from src.client.dataset_retriever.hf import HuggingFaceClient, DATASET_DIR
from src.chunker import Chunker
from include.utils import clear_directory

def process_single_dataset(
    dataset: str, local_fpath: str, dataset_client: AbstractDatasetClient
):
    """
    This processes 1 dataset, by downloading, chunking, embedding, and storing it
    in mongodb
    """

    if not os.path.exists(local_fpath):
        dataset_client.download_dataset(dataset_id=dataset, local_filepath=local_fpath)

    chunker = Chunker()
    dfiles = dataset_client.get_data_file_from_dir(local_fpath)
    print(f"Uploading dataset {dfiles}")

    try:
        chunks = chunker.process_files(dfiles, dataset)

        # Gathering other metadata for the document
        summary=dataset_client.generate_dataset_summary(local_fpath)
        links=dataset_client.generate_dataset_link(dataset)

        chunker.upload_chunks_to_mongo(chunks, links=links, dataset_summary=summary)
    except ValueError:
        print(f"Dataset {dataset} couldn't be read. Continuing...")


def main_loop(n_proc: int):
    """
    Main iteration loop. This runs constantly, pulling dataset after dataset, chunking

    n_proc: the number of processes to fire off the process_dataset function. TODO implement
    """

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
