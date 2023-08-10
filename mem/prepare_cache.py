import numpy as np
import torch
from tqdm import tqdm


model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
model = model.cuda()
model.eval()

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

# Specify the directory containing the images
# data_dir = "/data/ALG23/images"
# save_dir = "/data/ALG23/feats"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/data/ALG23/images")
parser.add_argument("--save_dir", type=str, default="/data/ALG23/feats")
args = parser.parse_args()
save_dir = args.save_dir
data_dir = args.data_dir

os.makedirs(save_dir, exist_ok=True)
# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
dataset = ImageDataset(data_dir, transform=transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    for batch in tqdm(dataloader):
        image = batch[0].cuda()
        feats = model(image)
        paths = batch[1]
        
        for i, path in enumerate(paths):
            path = path.split(".")[0] + ".npy"
            feat = feats[i].cpu().numpy().astype(np.float16)
            np.save(os.path.join(save_dir, path), feat)