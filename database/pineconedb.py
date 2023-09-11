import pinecone
import itertools
from tqdm import tqdm


class PineConeVDB:
    def __init__(self, api_key):
        self.api_key = api_key
        pinecone.init(      
            api_key=api_key,      
            environment='us-west1-gcp'      
        )               
        
    def create_index(self, index_name, dimension=512, metric="cosine"):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name, 
                dimension=dimension, 
                metric=metric
            )
            print(f"Index {index_name} created successfully!")

        else:
            print("Index already exists!")

    def insert_records(self, index_name, vector_data):
        index = pinecone.Index(index_name)
        index.upsert(
            vectors=vector_data
        )
        print("Inserted successfully")  

    def insert_batch(self, index_name, vectors_data):
        index = pinecone.Index(index_name)
        chunks_vectors_upsert = self.chunks(vectors_data, batch_size=100)
        for chunk_vectors in tqdm(chunks_vectors_upsert, desc="Inserting batch vectors"):
            index.upsert(
                vectors=chunk_vectors
            )
        print("Inserted successfully")

    def delete(self, index_name, vector_id):
        index = pinecone.Index(index_name)
        index.delete(ids=vector_id)
        print("Deleted successfully!")
        
    @staticmethod
    def chunks(iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))