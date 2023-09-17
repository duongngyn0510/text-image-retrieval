import torch


class AppConfig:
    INPUT_RESOLUTION = 224
    PRETRAINED_PATH = "./model_storage/pretrained_clip.pt"
    INDEX_NAME = "fashion"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "./data.csv"
    TOP_IMAGES = 8
    PORT_EXPOSE = 30000
