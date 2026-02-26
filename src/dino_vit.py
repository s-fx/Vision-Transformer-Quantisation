# PCA
# https://github.com/eriktaylor/Transformer-introduction/blob/main/DYNOv2_PCA.ipynb

import torch
import json
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

from PIL import Image
from torchvision.transforms import v2
from src.dinov2.models.vision_transformer import vit_base


class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        # Resize the image
        img = v2.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = v2.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)

        return img


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DinoVisionTransformerClassifier, self).__init__()
        model = vit_base(patch_size=14,
                         img_size=526,
                         init_values=1.0,
                         num_register_tokens=4,
                         block_chunks=0)
        self.embedding_size = 768
        self.number_of_heads = 12
        self.n_classes = n_classes
        ckpt = torch.load('/home/s-fx/fun/weights/dinov2_vitb14_reg4_pretrain.pth')
        model.load_state_dict(ckpt, strict=False)
        self.transformer = deepcopy(model)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 512), # was 256
            nn.BatchNorm1d(512), # new
            nn.ReLU(),
            nn.Linear(512, self.n_classes))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x



def visualise_features(model, image_path, device):
    class_id = image_path.split('/')[-2]
    print(class_id)
    model = model.transformer
    image = Image.open(image_path).convert('RGB')
    _, val_transforms = get_transforms()
    image_tensor = val_transforms(image)
    batch_size = 1
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features_dict = model.forward_features(image_tensor)
    features = features_dict['x_norm_patchtokens']
    _, n_patch, dim = features.shape
    features = features.reshape(batch_size*n_patch, dim)

    features = features.cpu()
    pca = PCA(n_components=1)
    scaler = MinMaxScaler()
    pca.fit(features)
    pca_features = pca.transform(features)
    norm_features = scaler.fit_transform(pca_features)

    grid_size = int(math.sqrt(n_patch))
    assert grid_size * grid_size == n_patch, "n_patch is not a perfect square"


    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs = np.atleast_1d(axs)

    # Undo normalization (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean).clip(0, 1)

    axs[0].imshow(img_np)
    axs[0].set_title(f"Original Image {class_id}")
    axs[0].axis("off")
    i = 0
    img = pca_features[i * n_patch: (i+1) * n_patch, 0].reshape(grid_size, grid_size)
    axs[1].imshow(img, cmap="viridis")
    axs[1].set_title("ViT Patch Features (PCA)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def get_transforms(image_dimension=256): # image dim could be 526

    # This is what DinoV2 sees
    target_size = (image_dimension, image_dimension)

    # Below are functions that every image will be passed through, including data augmentations
    train_transforms = v2.Compose([
                ResizeAndPad(target_size, 14),
                v2.RandomRotation(360),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    val_transforms = v2.Compose([
                ResizeAndPad(target_size, 14),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    return train_transforms, val_transforms


def load_dino(params_path, ckpt_path, device):
    with open(params_path, 'r') as f:
        p = json.load(f)

    number_of_classes = p['number_of_classes']
    model = DinoVisionTransformerClassifier(n_classes=number_of_classes)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model




