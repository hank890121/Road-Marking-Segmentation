import torch
import numpy as np

from engine.dataloader import ImgAnnDataset
from engine import transform
from engine.category import Category
from tqdm import tqdm

from engine.builder import build_model

categories = Category.load("./csv/rlmd.csv")
num_categories = Category.get_num_categories(categories)

# Load the pre-trained semantic segmentation model
model = build_model(
    {
        "name": "segformer",
        "pretrained": "nvidia/mit-b0",
        "num_classes": num_categories,
    }
).cuda()
model.load_state_dict(torch.load("/path/to/your/checkpoint")["model"])
model.eval()

transforms = [
    transform.LoadImg(),
    transform.LoadAnn(),
    transform.Resize((512, 512)),
    transform.Normalize(),
]

clear_dataloader = ImgAnnDataset(
    "/path/to/clear/val",
    transforms,
    "images",
    "labels",
    ".jpg",
    ".png",
).get_loader(1)

night_dataloader = ImgAnnDataset(
    "/path/to/night/val",
    transforms,
    "images",
    "labels",
    ".jpg",
    ".png",
).get_loader(1)

rainy_dataloader = ImgAnnDataset(
    "/path/to/rainy/val",
    transforms,
    "images",
    "labels",
    ".jpg",
    ".png",
).get_loader(1)

# Extract features
features_list = []
for i, data in enumerate(tqdm(clear_dataloader)):
    img = data["img"].cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.segformer(img, output_hidden_states=True).hidden_states[-1]
    features_list.append(features.cpu().numpy().flatten())

np.save("clear.npy", features_list)
features_list.clear()

for i, data in enumerate(tqdm(night_dataloader)):
    img = data["img"].cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.segformer(img, output_hidden_states=True).hidden_states[-1]
    features_list.append(features.cpu().numpy().flatten())

np.save("night.npy", features_list)
features_list.clear()

for i, data in enumerate(tqdm(rainy_dataloader)):
    img = data["img"].cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.segformer(img, output_hidden_states=True).hidden_states[-1]
    features_list.append(features.cpu().numpy().flatten())

np.save("rainy.npy", features_list)
features_list.clear()
