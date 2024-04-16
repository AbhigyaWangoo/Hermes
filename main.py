from huggingface_hub import hf_hub_download
import pandas as pd

if __name__ == "__main__":
    ds = "berkeley-nest/Nectar"
    FILENAME = "data/hf_dataset_nest.csv"

    dataset = pd.read_csv(
        hf_hub_download(repo_id=ds, filename=FILENAME, repo_type="dataset")
    )
