class Config:
    BATCH_SIZE = 32
    INPUT_RESOLUTION = 224
    TRAIN_EPOCHS = 5
    TRAIN_DATALOADER_PATH = "../notebooks/data_loader/train_dataloader.pkl"
    TEST_DATALOADER_PATH = "../notebooks/data_loader/test_dataloader.pkl"
    PRETRAINED_PATH = "./model_storage/pretrained_clip.pt"
    EXTRACT_DIR = "/tmp/GLAMI-1M/"
    EMBEDDING_PATH = "./embedding/embedding_images.json"
