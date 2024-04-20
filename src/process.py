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
        summary = dataset_client.generate_dataset_summary(local_fpath)
        links = dataset_client.generate_dataset_link(dataset)

        chunker.upload_chunks_to_mongo(chunks, links=links, dataset_summary=summary)
        clear_directory(local_fpath)
    except ValueError as e:
        print(f"Dataset {dataset} couldn't be read. Continuing... {e}")
        log_failed(dataset, e)
    except Exception as e:
        print(f"Dataset had some error, continuing like nothing happened... {e}")
        log_failed(dataset, e)


def log_failed(dataset: str, err: str):
    """Logs a failed dataset's name to file, along with the error."""
    logfile = "failed.txt"

    mode = "w"
    if os.path.exists(logfile):
        mode = "a"

    with open(logfile, mode, encoding="utf8") as fp:
        fp.write(f"{dataset} load failed, error: {err}\n")


def main_loop(n_proc: int):
    """
    Main iteration loop. This runs constantly, pulling dataset after dataset, chunking

    n_proc: the number of processes to fire off the process_dataset function. TODO implement
    """
    n_proc += 1  # to silence unused warning

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
