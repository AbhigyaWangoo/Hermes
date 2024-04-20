import os
from src.client.dataset_retriever.base import AbstractDatasetClient
from src.client.dataset_retriever.hf import HuggingFaceClient, DATASET_DIR
from src.chunker import Chunker
from multiprocessing import Process, Queue
import time
from typing import List


def get_data_file_from_dir(directory: str) -> List[str]:
    """
    Given a directory, this returns the path of all files within that directory
    (as well as nested files) that have 'train' or 'test' in their name. The
    output is a list of these paths.
    """
    data_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if "train" in file or "test" in file or "valid" in file:
                data_files.append(os.path.join(root, file))

    return data_files


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
    dfiles = get_data_file_from_dir(local_fpath)
    print(f"Uploading dataset {dfiles}")
    chunks = chunker.process_files(dfiles, dataset)

    chunker.upload_chunks_to_mongo(chunks)


def main_loop(n_proc: int):
    """
    Main iteration loop. This runs constantly, pulling dataset after dataset, chunking

    n_proc: the number of processes to fire off the process_dataset function.
    """

    # Queue for communication between processes
    dataset_queue = Queue()

    while True:
        # Retrieve datasets and add them to the queue
        dataset_client = HuggingFaceClient()
        datasets_to_process = dataset_client.list_datasets("text-classification")
        for dataset in datasets_to_process:
            dataset_queue.put(dataset)

        # Start the processing processes
        processes = []
        for _ in range(n_proc):
            if not dataset_queue.empty():
                dataset = dataset_queue.get()
                p = Process(
                    target=process_single_dataset,
                    args=(dataset, os.path.join(DATASET_DIR, dataset), dataset_client),
                )
                p.start()
                processes.append(p)
            else:
                break

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Sleep for some time before checking for new datasets again
        time.sleep(10)  # Adjust sleep time as needed
