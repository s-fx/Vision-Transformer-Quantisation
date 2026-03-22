import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dino_vit import load_dino, visualise_features, get_transforms
from src.dataset.dataset import RetinopathyFullDataset, CIFAR10Dataset
from torch.utils.data import DataLoader
from pathlib import Path

output_dir = Path("vit_activation_analysis")
output_dir.mkdir(exist_ok=True)

activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook


def register_hooks(model):

    for i, block in enumerate(model.transformer.blocks):

        block.norm1.register_forward_hook(
            get_hook(f"block{i}_norm1")
        )

        block.attn.qkv.register_forward_hook(
            get_hook(f"block{i}_qkv")
        )

        block.mlp.fc1.register_forward_hook(
            get_hook(f"block{i}_mlp")
        )

def plot_surface(data, title, path):

    tokens, channels = data.shape

    X, Y = np.meshgrid(
        np.arange(channels),
        np.arange(tokens)
    )

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        data,
        cmap="coolwarm",
        linewidth=0
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlabel("Magnitude")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_channel_std(data, title, path):

    std = data.std(axis=0)

    plt.figure(figsize=(6,3))
    plt.plot(std)

    plt.xlabel("Channel")
    plt.ylabel("Standard Deviation")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_activation_histogram(tensor, name, path):

    data = tensor.flatten()

    plt.figure(figsize=(6,4))
    plt.hist(data, bins=200)

    plt.title(f"Activation Distribution: {name}")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_channel_range(data, name, path):

    channel_max = np.max(data, axis=0)
    channel_min = np.min(data, axis=0)

    channel_range = channel_max - channel_min

    plt.figure(figsize=(6,3))
    plt.plot(channel_range)

    plt.title(f"Channel Range: {name}")
    plt.xlabel("Channel")
    plt.ylabel("Activation Range")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_outliers(data, name, path):

    max_vals = np.max(np.abs(data), axis=0)

    plt.figure(figsize=(6,3))
    plt.plot(max_vals)

    plt.title(f"Activation Outliers per Channel: {name}")
    plt.xlabel("Channel")
    plt.ylabel("Max |Activation|")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_layer_dynamic_range(results):

    layers = list(results.keys())
    ranges = list(results.values())

    plt.figure(figsize=(8,4))

    plt.bar(range(len(layers)), ranges)

    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.ylabel("Dynamic Range")

    plt.title("Activation Dynamic Range Across Layers")

    plt.tight_layout()
    plt.savefig(output_dir / "layer_dynamic_range.png")
    plt.close()


def analyze_vit(model, dataloader, device):

    model.eval()

    register_hooks(model)

    images, labels = next(iter(dataloader))
    image = images[0].unsqueeze(0).to(device)

    with torch.no_grad():
        model(image)

    dynamic_ranges = {}

    for name, tensor in activations.items():

        data = tensor[0].abs().numpy()

        # Surface
        plot_surface(
            data,
            name,
            output_dir / f"{name}_surface.png"
        )

        # Histogram
        plot_activation_histogram(
            data,
            name,
            output_dir / f"{name}_hist.png"
        )

        # Channel Range
        plot_channel_range(
            data,
            name,
            output_dir / f"{name}_range.png"
        )

        # Outliers
        plot_outliers(
            data,
            name,
            output_dir / f"{name}_outliers.png"
        )

        # Channel STD
        plot_channel_std(
            data,
            name,
            output_dir / f"{name}_std.png"
        )


        # Dynamic Range
        dynamic_ranges[name] = np.max(data) - np.min(data)

    plot_layer_dynamic_range(dynamic_ranges)


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
    model.to(device)

    analyze_vit(model, dataloader, device)


if __name__ == '__main__':
    main()
