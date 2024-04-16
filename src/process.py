import os
from src.client.dataset_retriever.base import AbstractDatasetClient
from src.client.dataset_retriever.hf import HuggingFaceClient, DATASET_DIR
from src.chunker import Chunker
from multiprocessing import Process, Queue
import time


def process_single_dataset(
    dataset: str, local_fpath: str, dataset_client: AbstractDatasetClient
):
    """
    This processes 1 dataset, by downloading, chunking, embedding, and storing it
    in mongodb
    """

    dataset_client.download_dataset(dataset_id=dataset, local_filepath=local_fpath)

    chunker = Chunker()
    chunks = chunker.process_file(local_fpath) # TODO currently this function is broken because it tries to read the folder as a csv.

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
