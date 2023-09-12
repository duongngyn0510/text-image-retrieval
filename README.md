# Build end2end text-image retrieval app

## Model

Fine-tune [CLIP](https://arxiv.org/abs/2103.00020) in image-retrieval task

+  **Input**: Image or text query.

+  **Output**: Top images with the highest similarity according to the cosine metrics.

## Database

Using [Pinecone](https://www.pinecone.io/) vector database for fast retrieval result
+ Vector database contains **85577** vector ids, those vectors are images embedding and their metadata.
