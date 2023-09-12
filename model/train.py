# Need training on Colab
import clip
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from loguru import logger
from config import Config
from custom_dataset import FashionDataset, BalancedBatchSampler
from utils import load_dataloader, convert_models_to_fp32

warnings.filterwarnings("ignore")

BATCH_SIZE = Config.BATCH_SIZE
TRAIN_DATALOADER_PATH = Config.TRAIN_DATALOADER_PATH
TEST_DATALOADER_PATH =  Config.TEST_DATALOADER_PATH
TRAIN_EPOCHS = Config.TRAIN_EPOCHS
PRETRAINED_PATH = Config.PRETRAINED_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
logger.info(f"Device: {device}")

loaded_train_dataloader = load_dataloader(TRAIN_DATALOADER_PATH)
loaded_test_dataloader = load_dataloader(TEST_DATALOADER_PATH)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

for epoch in range(1, TRAIN_EPOCHS + 1):
    pbar = tqdm(loaded_train_dataloader, total=len(loaded_train_dataloader))
    train_loss = 0
    for batch in pbar:
        optimizer.zero_grad()

        image_tensor, prompt_token, _ = batch

        image_tensor = image_tensor.to(device)
        prompt_token = prompt_token.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(image_tensor, prompt_token)
        ground_truth = torch.arange(BATCH_SIZE).to(device)

        # Compute loss
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        # Backward pass
        if device == "cpu":
            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        else:
            total_loss.backward()
            train_loss += total_loss
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{TRAIN_EPOCHS}, Loss: {total_loss.item():.4f}")

    test_loss = 0
    with torch.no_grad():
        for batch in loaded_test_dataloader:
            image_tensor, prompt_token, _ = batch

            image_tensor = image_tensor.to(device)
            prompt_token = prompt_token.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(image_tensor, prompt_token)
            ground_truth = torch.arange(BATCH_SIZE).to(device)

            # Compute loss
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            test_loss += total_loss

    print(f"Epoch {epoch}, Train loss {train_loss / len(loaded_train_dataloader)}")
    print(f"Epoch {epoch}, Train loss {test_loss / len(loaded_test_dataloader)}")

torch.save(model, PRETRAINED_PATH)
logger.info(f"Model saved successfully in {PRETRAINED_PATH}!")
