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
        for filename_idx in embeddings:
            idx = filename_idx
            embedding = embeddings[filename_idx]

            document = {
                "_id": idx,  # filename_chunkindex
                "dataset_embedding": embedding,  # Assuming embeddings are lists of lists
            }

            document.update(kwargs)  # Add additional fields from kwargs
            self.collection.insert_one(document)

    def delete_all(self):
        """Deletes an entire collection. Be careful with this one..."""
        self.collection.delete_many({})

    def perform_vector_search(self, query_embedding):
        # define pipeline
        pipeline = [
            {
                '$vectorSearch': {
                'index': 'vector_index', 
                'path': 'dataset_embedding', 
                'queryVector': query_embedding,
                'numCandidates': 50, 
                'limit': 5
                }
            }, {
                '$project': {
                '_id': 1, 
                'links': 1, 
                'dataset_embedding': 1, 
                'dataset_summary': 1, 
                'score': {
                    '$meta': 'vectorSearchScore'
                }
                }
            }
        ]

        print("pipeline constructed")
        result = self.client[self.database_name][self.collection_name].aggregate(pipeline)

        print("running pipeline finished")

        # print(list(result))
        return result 
 
    def delete_all(self):
        """Deletes an entire collection. Be careful with this one..."""
        self.collection.delete_many({})
