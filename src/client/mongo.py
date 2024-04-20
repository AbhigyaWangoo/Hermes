import pymongo
from typing import Dict, List

class MongoDBUploader:
    """A helper class to upload mongo db embeddings"""

    def __init__(self, connection_string, database_name, collection_name):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def upload_embeddings(self, embeddings: Dict[str, List[float]], **kwargs):
        """
        Upload embeddings to MongoDB Atlas.

        embeddings: A dictionary of {"datasetname_chunkidx: [embeddings]"}
        """
        for idx, embedding in enumerate(embeddings):
            document = {
                "_id": idx,  # filename_chunkindex
                "dataset_embedding": embedding,  # Assuming embeddings are lists of lists
            }
            document.update(kwargs)  # Add additional fields from kwargs
            self.collection.insert_one(document)
