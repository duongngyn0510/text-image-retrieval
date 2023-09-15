import json
import os
import sys

from dotenv import dotenv_values

sys.path.append(os.path.normpath(os.path.join("../")))

from database.pineconedb import PineConeVDB

config = dotenv_values("../database/.env")
api_key = config["API_KEY"]

index_name = "fashion"
pineconevdb = PineConeVDB(api_key)
pineconevdb.create_index(index_name)

with open("embedding/embedding_images.json", "r") as f:
    embedding_data = json.load(f)

# len(embedding_data) = 85577
# n vectors need to be inserted to Pinecone DB is 85577
pineconevdb.insert_batch(index_name, embedding_data)
