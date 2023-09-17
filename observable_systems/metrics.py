from io import BytesIO
from time import time

import imagehash
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from PIL import Image
from prometheus_client import start_http_server
from utils.config import AppConfig
from utils.pinecone_utils import get_index, search
from utils.utils import *

start_http_server(port=8099, addr="0.0.0.0")

# Service name is required for most backends
resource = Resource(attributes={SERVICE_NAME: "retrieval-service"})

# Exporter to export metrics to Prometheus
reader = PrometheusMetricReader()

# Meter is responsible for creating and recording metrics
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)
meter = metrics.get_meter("my_retrieval", "0.1.2")

counter = meter.create_counter(
    name="app_request_counter", description="Number of app requests"
)

histogram = meter.create_histogram(
    name="app_response_histogram",
    description="app response histogram",
    unit="seconds",
)


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


@app.post("/image_url")
async def image_url(image_file: UploadFile = File(...)):
    """Get image url with image file query

    Args:
        image_file (UploadFile)

    Returns:
        (List): List of top images url
    """
    request_image_content = await image_file.read()
    pil_image = Image.open(BytesIO(request_image_content))
    pil_hash = imagehash.average_hash(pil_image)
    logger.info(pil_hash)

    if pil_hash not in image_cache:
        logger.info("Getting related products!")
        image_embedding = get_image_embedding(model, DEVICE, pil_image)
        match_ids = search(index, image_embedding, top_k=AppConfig.TOP_IMAGES)
        image_cache[pil_hash] = match_ids
    else:
        logger.info("Getting related products from cache!")
        match_ids = image_cache[pil_hash]

    images_url = get_image_url(
        match_ids,
    )
    return images_url


@app.post("/display_image")
async def display_image(image_file: UploadFile = File(...)):
    """Display images from their urls with image file query

    Args:
        image_file (UploadFile)

    Returns:
        HTMLResponse
    """
    starting_time = time()
    images_url = await image_url(image_file)
    html_content = display_html(images_url)
    # Increase the counter
    label = {"api": "/display_image"}

    counter.add(1, label)

    # Mark the end of the response
    ending_time = time()
    elapsed_time = ending_time - starting_time

    # Add histogram
    logger.info("elapsed time: ", elapsed_time)
    logger.info(elapsed_time)
    histogram.record(elapsed_time, label)

    return HTMLResponse(content=html_content)


@app.post("/text_url")
async def text_url(text_query: str):
    """Get image url with text query

    Args:
        text_query (str)

    Returns:
       (List): List of top images url
    """

    if text_query not in text_cache:
        logger.info("Getting related products!")
        text_embedding = get_text_embedding(model, DEVICE, text_query)
        match_ids = search(index, text_embedding, top_k=AppConfig.TOP_IMAGES)
        text_cache[text_query] = match_ids
    else:
        logger.info("Getting related products from cache!")
        match_ids = text_cache[text_query]

    images_url = get_image_url(match_ids)
    return images_url


@app.post("/display_text")
async def display_text(text_query: str):
    """Display images from their urls with text query

    Args:
        text_query (str)

    Returns:
        HTMLResponse
    """

    starting_time = time()
    images_url = await text_url(text_query)
    html_content = display_html(images_url)
    label = {"api": "/display_text"}

    # Increase the counter
    counter.add(1, label)

    # Mark the end of the response
    ending_time = time()
    elapsed_time = ending_time - starting_time

    # Add histogram
    logger.info("elapsed time: ", elapsed_time)
    logger.info(elapsed_time)
    histogram.record(elapsed_time, label)

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
