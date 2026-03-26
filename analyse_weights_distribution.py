import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.dino_vit import load_dino

output_dir = Path("vit_weight_analysis")
output_dir.mkdir(exist_ok=True)


def get_linear_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers[name] = module
    return layers


def plot_histogram(data, title, path):
    plt.figure(figsize=(6,4))
    plt.hist(data.flatten(), bins=200)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_channel_range(weight, title, path):
    # per output channel (rows)
    w = weight.detach().cpu().numpy()
    channel_max = np.max(w, axis=1)
    channel_min = np.min(w, axis=1)
    channel_range = channel_max - channel_min

    plt.figure(figsize=(6,3))
    plt.plot(channel_range)
    plt.title(title)
    plt.xlabel("Output Channel")
    plt.ylabel("Range")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return channel_range


def plot_outliers(weight, title, path):
    w = weight.detach().cpu().numpy()
    max_vals = np.max(np.abs(w), axis=1)

    plt.figure(figsize=(6,3))
    plt.plot(max_vals)
    plt.title(title)
    plt.xlabel("Output Channel")
    plt.ylabel("Max |Weight|")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def analyze_weights(model):

    layers = get_linear_layers(model)

    range_results = {}

    for name, layer in layers.items():

        weight = layer.weight

        print(f"[INFO] Analyzing layer: {name} | shape: {weight.shape}")

        # Histogram
        plot_histogram(
            weight.detach().cpu().numpy(),
            f"Weight Distribution: {name}",
            output_dir / f"{name}_hist.png"
        )

        # Channel range
        channel_range = plot_channel_range(
            weight,
            f"Channel Range: {name}",
            output_dir / f"{name}_range.png"
        )

        range_results[name] = np.mean(channel_range)

        # Outliers
        plot_outliers(
            weight,
            f"Outliers: {name}",
            output_dir / f"{name}_outliers.png"
        )

    return range_results


def plot_layer_metric(results, title, ylabel, filename):

    layers = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10,4))
    plt.bar(range(len(layers)), values)
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()



def main():

    root = './runs/run_dino_retino_no_backbone'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_32_model.pth')
    device = 'cuda'

    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    model.to(device)

    range_results = analyze_weights(model)

    # Plot summaries
    plot_layer_metric(
        quant_results,
        "Weight Quantization Sensitivity per Layer",
        "MSE",
        "weight_quant_mse.png"
    )

    plot_layer_metric(
        range_results,
        "Average Channel Range per Layer",
        "Range",
        "weight_range.png"
    )

    print("\n[RESULTS] Avg Channel Range:")
    for k, v in range_results.items():
        print(k, v)


if __name__ == "__main__":
    main()
