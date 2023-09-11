import os
import load_dataset
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
import warnings
warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
logger.info("Loaded model successfully!")

pretrained_model = torch.load('model_storage/pretrained_clip.pt')
logger.info("Loaded pretrained model successfully!")

os.environ['EXTRACT_DIR'] = "/tmp/GLAMI-1M/"
load_dataset.download_dataset(dataset_url="https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip")
logger.info("Loaded dataset successfully!")

df = load_dataset.get_dataframe('test')[['category_name', 'image_file']]
df_compare = df.sample(10, random_state=41)

def visualize_similarity(using_pretrained=True):

    image_embeddings = []
    text_embeddings = []
    images = []
    texts = []

    for row in df_compare.itertuples(index=True):

        text = row.category_name
        texts.append(text)

        image = Image.open(row.image_file).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).cuda()
        images.append(image)

        with torch.no_grad():
            if using_pretrained:
                image_embedding = pretrained_model.encode_image(image_input).float()
                text_embedding = pretrained_model.encode_text(clip.tokenize(text).to(device)).float()
            else:
                image_embedding = model.encode_image(image_input).float()
                text_embedding = model.encode_text(clip.tokenize(text).to(device)).float()
                
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        image_embeddings.append(image_embedding.cpu().numpy())
        text_embeddings.append(text_embedding.cpu().numpy())
        
    similarity = np.zeros((len(images), len(images)), dtype=float)
    for i, image_embedding in enumerate(image_embeddings):
        for j, text_embedding in enumerate(text_embeddings):
            dot = image_embedding @ text_embedding.T
            similarity[i, j] = dot
            
    count = len(texts)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title(f"Cosine similarity between text and image embeddings {'using' if using_pretrained is True else 'not using'} pretrained model", size=20)
    plt.savefig(f"text_image_similarity/{'using' if using_pretrained is True else 'not_using'}_pretrained_model.png",  bbox_inches='tight')
    plt.show()
    

visualize_similarity(using_pretrained=False)
visualize_similarity(using_pretrained=True)