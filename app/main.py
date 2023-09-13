import base64
from io import BytesIO

import imagehash
import requests
import torch
from config import AppConfig
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from pinecone_utils import get_index, search
from utils import get_image_embedding, get_image_url, get_text_embedding

INDEX_NAME = AppConfig.INDEX_NAME
index = get_index(INDEX_NAME)
logger.info(f"Connect to index {INDEX_NAME} in Pinecone successfully!")

PRETRAINED_PATH = AppConfig.PRETRAINED_PATH
DEVICE = AppConfig.DEVICE
model = torch.load(PRETRAINED_PATH, map_location=torch.device(DEVICE))
model.eval()

if DEVICE == "cpu":
    for p in model.parameters():
        p.data = p.data.float()

logger.info("Load pretrained model successfully!")

app = FastAPI()
image_cache = {}
text_cache = {}


@app.post("/image")
async def image(image_file: UploadFile = File(...)):
    request_image_content = await image_file.read()
    pil_image = Image.open(BytesIO(request_image_content))
    pil_hash = imagehash.average_hash(pil_image)
    logger.info(pil_hash)

    if pil_hash not in image_cache:
        logger.info("Getting related products!")
        image_embedding = get_image_embedding(model, DEVICE, pil_image)
        match_ids = search(index, image_embedding, top_k=5)
        image_cache[pil_hash] = match_ids
    else:
        logger.info("Getting related products from cache!")
        match_ids = image_cache[pil_hash]

    images_url = get_image_url(match_ids)
    encoded_images = []
    for image_url in images_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                encoded_image = base64.b64encode(response.content).decode("utf-8")
                encoded_images.append(encoded_image)
            else:
                encoded_images.append(None)

        except Exception as e:
            encoded_images.append(None)
    return {"images": encoded_images}


@app.post("/text")
async def text(text_query: str):
    if text_query not in text_cache:
        logger.info("Getting related products!")
        text_embedding = get_text_embedding(model, DEVICE, text_query)
        match_ids = search(index, text_embedding, top_k=5)
        text_cache[text_query] = match_ids
    else:
        logger.info("Getting related products from cache!")
        match_ids = image_cache[text_query]

    images_url = get_image_url(match_ids)
    return images_url
