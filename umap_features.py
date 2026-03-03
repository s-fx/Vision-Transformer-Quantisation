import os
import glob
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from src.dino_vit import load_dino, visualise_features, get_transforms
from src.dataset.dataset import RetinopathyFullDataset, CIFAR10Dataset
from torch.utils.data import DataLoader



reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=2,
    metric="cosine"  # better for transformer embeddings
)


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            outputs = model(images)

            cls_embeddings = model.transformer.forward_features(images)['x_norm_clstoken']

            features.append(cls_embeddings.cpu())
            labels.append(targets)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels


def main():
    root = './runs/run_dino_cifar_backbone'
    data_root = '/home/s-fx/fun/datasets/CIFAR-10-dataset'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_2_model.pth')
    loss_dict = os.path.join(root, 'loss_dict.pkl')
    device = 'cuda'
    #images_path = glob.glob(f'{data_root}/single_example/*/*.jpg')
    _, val_transform = get_transforms()

    dataset = CIFAR10Dataset(data_root, val_transform, mode='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # ------------ BASE MODEL --------------
    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    model = torch.compile(model, mode='max-autotune')
    model.to(device)

    features, labels = extract_features(model, dataloader, device)
    embedding_2d = reducer.fit_transform(features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap='tab10',
        s=5
    )

    plt.colorbar(scatter)
    plt.title("UMAP Projection of ViT Feature Space")
    plt.show()

if __name__ == '__main__':
    main()
