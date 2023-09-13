import pinecone
from dotenv import dotenv_values

config = dotenv_values("../database/.env")
api_key = config["API_KEY"]


def get_index(index_name):
    pinecone.init(api_key=api_key, environment="us-west1-gcp")
    index = pinecone.Index(index_name)
    return index


def search(index, input_emb, top_k):
    matching = index.query(vector=input_emb, top_k=top_k, include_values=True)[
        "matches"
    ]
    match_ids = [match_id["id"] for match_id in matching]
    return match_ids
