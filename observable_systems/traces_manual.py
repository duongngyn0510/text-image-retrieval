from io import BytesIO
from time import time

import imagehash
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider
from PIL import Image
from utils.config import AppConfig
from utils.pinecone_utils import get_index, search
from utils.utils import *

set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "retrieval-service"}))
)
tracer = get_tracer_provider().get_tracer("myretrieval", "0.1.2")
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
get_tracer_provider().add_span_processor(span_processor)


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


@app.post("/display_image")
async def image_url(image_file: UploadFile = File(...)):
    """Display images from their urls with image file query

    Args:
        image_file (UploadFile)

    Returns:
        HTMLResponse
    """
    with tracer.start_as_current_span("processors") as processors:
        with tracer.start_as_current_span(
            "get-image", links=[trace.Link(processors.get_span_context())]
        ):
            request_image_content = await image_file.read()
            pil_image = Image.open(BytesIO(request_image_content))
            pil_hash = imagehash.average_hash(pil_image)
            logger.info(pil_hash)

        if pil_hash not in image_cache:
            with tracer.start_as_current_span(
                "preprocess-and-get-image-embedding",
                links=[trace.Link(processors.get_span_context())],
            ):
                logger.info("Getting related products!")
                image_embedding = get_image_embedding(model, DEVICE, pil_image)

            with tracer.start_as_current_span(
                "query-image-embedding",
                links=[trace.Link(processors.get_span_context())],
            ):
                match_ids = search(index, image_embedding, top_k=AppConfig.TOP_IMAGES)
                image_cache[pil_hash] = match_ids
        else:
            logger.info("Getting related products from cache!")
            match_ids = image_cache[pil_hash]

        with tracer.start_as_current_span(
            "get-image-url", links=[trace.Link(processors.get_span_context())]
        ):
            images_url = get_image_url(
                match_ids,
            )
        html_content = display_html(images_url)

    return HTMLResponse(content=html_content)


@app.post("/display_text")
async def text_url(text_query: str):
    """Display images from their urls with text query

    Args:
        text_query (str)

    Returns:
        HTMLResponse
    """
    with tracer.start_as_current_span("processors") as processors:
        if text_query not in text_cache:
            with tracer.start_as_current_span(
                "get-text-embedding", links=[trace.Link(processors.get_span_context())]
            ):
                logger.info("Getting related products!")
                text_embedding = get_text_embedding(model, DEVICE, text_query)
            with tracer.start_as_current_span(
                "query-image-embedding",
                links=[trace.Link(processors.get_span_context())],
            ):
                match_ids = search(index, text_embedding, top_k=AppConfig.TOP_IMAGES)
                text_cache[text_query] = match_ids
        else:
            logger.info("Getting related products from cache!")
            match_ids = text_cache[text_query]

        with tracer.start_as_current_span(
            "get-image-url", links=[trace.Link(processors.get_span_context())]
        ):
            images_url = get_image_url(match_ids)
        html_content = display_html(images_url)

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
