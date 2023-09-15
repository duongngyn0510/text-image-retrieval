import warnings

import torch
from config import Config
from custom_dataset import FashionInfer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

warnings.filterwarnings("ignore")

PRETRAINED_PATH = Config.PRETRAINED_PATH
EMBEDDING_PATH = Config.EMBEDDING_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

df_infer = get_dataset()
logger.info("Loaded dataset successfully!")

model = torch.load(PRETRAINED_PATH)
logger.info("Loaded pretrained model successfully!")

images_list = []
for image in tqdm(df_infer.image_file, desc="Appending images"):
    images_list.append(Image.open(image).convert("RGB"))

dataset = FashionInfer(images_list)
data_loader = DataLoader(dataset, batch_size=16)

# perform batch inference for create images embedding
embedding_images = []
model.eval()
for images_tensor in tqdm(data_loader, desc="Inferencing"):
    images_tensor = images_tensor.to(device)
    embedding_image = model.encode_image(images_tensor)
    embedding_images.extend(
        [i.tolist() for i in embedding_image.detach().cpu().numpy()]
    )

vectors_upsert = []
for i, (product_name, embedding) in enumerate(
    zip(df_infer.category_name.values, embedding_images)
):
    vector_upsert = {
        "id": str(i),
        "values": embedding,
        "metadata": {"product name": product_name},
    }
    vectors_upsert.append(vector_upsert)

# save embedding data in json file
save_embedding(EMBEDDING_PATH, vectors_upsert)
