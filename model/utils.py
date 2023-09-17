import json
import os
import pickle

import load_dataset
from config import Config
from loguru import logger
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

BICUBIC = InterpolationMode.BICUBIC
INPUT_RESOLUTION = Config.INPUT_RESOLUTION
EXTRACT_DIR = Config.EXTRACT_DIR


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def preprocess(n_px=INPUT_RESOLUTION):
    """
    A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input

    Args:
        n_px : int, default=224
        Input resolution

    Returns:
        preprocess : Callable[[PIL.Image], torch.Tensor]
    """
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load_dataloader(path):
    """
    Load dataloader from pkl file

    Args:
        path (str) : Directory to dataloader fkl file

    Returns:
        torch.utils.data.DataLoader
    """
    with open(path, "rb") as f:
        loaded_dataloader = pickle.load(f)
    logger.info(f"Loaded {path.split('/')[-1].split('_')[0]} dataloader successfully!")
    return loaded_dataloader


def get_dataset():
    """
    Return small GLAMI-1M-dataset in DataFrame form

    Returns:
        pandas.DataFrame
    """
    os.environ["EXTRACT_DIR"] = EXTRACT_DIR
    load_dataset.download_dataset(
        dataset_url="https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip"
    )
    df = load_dataset.get_dataframe("test")[["category_name", "image_file"]]
    df = df.drop_duplicates(subset=["image_file"])[
        ["category_name", "image_file"]
    ].reset_index(drop=True)
    return df


def save_embedding(path, vectors_upsert):
    """
    Save embedding images and their other data in json file

    Args:
        path (str): directory to save json file
        vectors_upsert (List[Dict]): vectors to upsert to Pinecone vector database
    """
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump(vectors_upsert, f)
    else:
        with open(path, "r") as f:
            values = json.load(f)
        values.extend(vectors_upsert)

        with open(path, "w") as f:
            json.dump(values, f)

    logger.info(f"Saved embedding successfully in {path}!")
