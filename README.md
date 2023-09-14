# Build end2end text-image retrieval app

## Requirements

```bash
$ pip install git+https://github.com/openai/CLIP.git
$ pin install -r requirements.txt
```

## Model

Fine-tune [CLIP](https://arxiv.org/abs/2103.00020) in image-retrieval task

+  **Input**: Image or text query related to *FASHION*.

+  **Output**: Top images with the highest similarity according to the cosine metrics.

## Database

Using [Pinecone](https://www.pinecone.io/) vector database for fast retrieval result
+ Vector database contains **85577** vector ids, those vectors are images embedding and their metadata.


## Local test
Can build the new image from `Dockerfile` or reuse my image.
```bash
$ docker pull duong05102002/text-image-retrieval:v1.0.0
```
Run `client.py` for test the api.

+ image query
```bash
$ python client.py --save_dir temp.html --image_query your_image_file
```
+ text query
```bash
$ python client.py --save_dir temp.html --text_query your_text_query
```
**Note**: Refresh the html page to display the images
