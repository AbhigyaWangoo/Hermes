from src.responder import QueryHandler
from src.client.dataset_retriever.kaggle import KaggleDatasetClient

if __name__ == "__main__":
    client = KaggleDatasetClient()

    print(client.list_datasets(""))