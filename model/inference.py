import os
import load_dataset
import clip
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from loguru import logger


device = "cuda" if torch.cuda.is_available() else "cpu"
_, preprocess = clip.load("ViT-B/32", device=device)

os.environ['EXTRACT_DIR'] = "/tmp/GLAMI-1M/"
load_dataset.download_dataset(dataset_url="https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip")


df_infer = load_dataset.get_dataframe('test')[['category_name', 'image_file']]
df_infer = df_infer.drop_duplicates(subset=['image_file'])[['category_name', 'image_file']].reset_index(drop=True)
# 16 of ram is not enough for the entire images after preprocess
# so only takes 30000 images 

sub_df_infer = df_infer[:30000]
# sub_df_infer = df_infer[30000: 60000]
# sub_df_infer = df_infer[60000:]

# load pretrained model
model = torch.load('model_storage/pretrained_clip.pt')
logger.info("Loaded pretrained model successfully!")

images_list = []
for image in tqdm(sub_df_infer.image_file, desc="Appending images"):
    images_list.append(Image.open(image).convert('RGB'))


class FashionInfer(Dataset):
    def __init__(self, images_list):
        self.images_list = images_list
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        image_tensor = preprocess(images_list[idx])
        return image_tensor
    
dataset = FashionInfer(images_list)
data_loader = DataLoader(dataset, batch_size=16)

# perform batch inference for create images embedding, it may takes 15 minutes 
embedding_images = []

model.eval()
for images_tensor in tqdm(data_loader, desc="Inferencing"):
    images_tensor = images_tensor.to(device)
    embedding_image = model.encode_image(images_tensor)
    embedding_images.extend([i.tolist() for i in embedding_image.detach().cpu().numpy()])

vectors_upsert = []
for i, (product_name, embedding) in enumerate(zip(sub_df_infer.category_name.values, embedding_images)):
    vector_upsert = {
        'id': str(i),
        'values': embedding,
        'metadata': {'product name': product_name}
    }
    vectors_upsert.append(vector_upsert)

# save embedding in json file
if not os.path.isfile('embedding/embedding_images.json'):
    with open('embedding/embedding_images.json', 'w') as f:
        json.dump(vectors_upsert, f)
else:
    with open('embedding/embedding_images.json', 'r') as f:
        values = json.load(f)  
    values.extend(vectors_upsert)

    with open('embedding/embedding_images.json', 'w') as f:
        json.dump(values, f)

logger.info("Saved successfully")