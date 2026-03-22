import os
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dino_vit import load_dino, get_transforms
from src.dataset.dataset import RetinopathyFullDataset
from src.quant.quantLayer_sym import replace_linear_layers, QuantizedLinearLayer


reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42  # wichtig!
)


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            cls_embeddings = model.transformer.forward_features(images)['x_norm_clstoken']

            features.append(cls_embeddings.cpu())
            labels.append(targets)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels


def main():
    root = './runs/run_dino_retino_no_backbone'
    data_root = '/home/s-fx/fun/datasets/retinopathy-full-ds-cleaned'
    device = 'cuda'

    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_32_model.pth')

    _, val_transform = get_transforms()

    dataset = RetinopathyFullDataset(data_root, val_transform, mode='test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ------------ FP32 MODEL --------------
    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    model.to(device)

    # ------------ QUANTIZED MODEL ----------
    model_quant = load_dino(params_path, ckpt_path, device)
    replace_linear_layers(model_quant, QuantizedLinearLayer, [''], quantized=True)
    model_quant.eval()
    model_quant.to(device)

    print("Extracting FP32 features...")
    features_fp32, labels = extract_features(model, dataloader, device)

    print("Extracting quantized features...")
    features_quant, _ = extract_features(model_quant, dataloader, device)

    # ------------ COMBINE -----------------
    combined_features = np.concatenate([features_fp32, features_quant], axis=0)

    class_labels = np.concatenate([labels, labels], axis=0)

    domain_labels = np.array(
        [0] * len(features_fp32) + [1] * len(features_quant)
    )  # 0 = FP32, 1 = Quantized

    print("Running UMAP...")
    embedding_2d = reducer.fit_transform(combined_features)

    # ------------ PLOT 1: CLASS STRUCTURE ----------
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=class_labels,
        cmap='tab10',
        s=5
    )
    plt.colorbar(scatter)
    plt.title("UMAP - Class Structure (FP32 + Quantized)")
    plt.show()

    # ------------ PLOT 2: DOMAIN SHIFT ----------
    plt.figure(figsize=(8, 8))

    plt.scatter(
        embedding_2d[domain_labels == 0, 0],
        embedding_2d[domain_labels == 0, 1],
        label="FP32",
        s=5,
        alpha=0.5
    )

    plt.scatter(
        embedding_2d[domain_labels == 1, 0],
        embedding_2d[domain_labels == 1, 1],
        label="Quantized",
        s=5,
        alpha=0.5,
        marker='x'
    )

    plt.legend()
    plt.title("UMAP - FP32 vs Quantized Feature Space")
    plt.show()

    # ------------ OPTIONAL: FEATURE DRIFT ----------
    cos_sim = np.sum(features_fp32 * features_quant, axis=1) / (
        np.linalg.norm(features_fp32, axis=1) *
        np.linalg.norm(features_quant, axis=1)
    )

    print("Mean cosine similarity:", cos_sim.mean())


if __name__ == '__main__':
    main()
