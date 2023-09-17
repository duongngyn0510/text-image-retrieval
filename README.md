# Build end2end text-image retrieval app

## Local service

### Requirements

```bash
$ pip install git+https://github.com/openai/CLIP.git
$ pin install -r requirements.txt
```

### Model

Fine-tune [CLIP](https://arxiv.org/abs/2103.00020) in image-retrieval task

+  **Input**: Image or text query related to *FASHION*.

+  **Output**: Top images with the highest similarity according to the cosine metrics.

### Database

Using [Pinecone](https://www.pinecone.io/) vector database for fast retrieval result
+ Vector database contains **85577** vector ids, those vectors are images embedding and their metadata.

Using Google Cloud Storage for storing image data

### Local test
```bash
$ docker pull duong05102002/retrieval-local-service:v1.23
$ docker run -p 30000:30000 duong05102002/retrieval-local-service:v1.23
```
Run `client.py` for test the local api.

+ Image query
```bash
$ python client.py --save_dir temp.html --image_query your_image_file
```
+ Text query
```bash
$ python client.py --save_dir temp.html --text_query your_text_query
```
**Note**: Refresh the html page to display the images

### Response time (traces)

+ Image query
![result](observable_systems/traces_image_query.png)

+ Text query
![result](observable_systems/traces_text_query.png)

**Note**: Refresh the html page to display the images

**Top 8 products images similar with image query:** 

![](app/images/woman_blazers.png)

<html>
    <body>
        <div class="image-grid">
<img src="https://storage.googleapis.com/fashion_image/168125.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/510624.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/919453.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/509864.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/1002845.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/6678.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/589519.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/67591.jpg" alt="Image" width="200" height="300">
        </body>
    </html>


**Top 8 products images similar with text query: crop top** 
<html>
    <body>
        <div class="image-grid">
<img src="https://storage.googleapis.com/fashion_image/640366.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/965820.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/607634.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/673682.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/615135.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/38530.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/455345.jpg" alt="Image" width="200" height="300"><img src="https://storage.googleapis.com/fashion_image/742095.jpg" alt="Image" width="200" height="300">
        </body>
    </html>
    
### CI/CD
`Jenkinsfile` for test CI/CD in local
