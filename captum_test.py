"""
https://captum.ai/tutorials/CIFAR_TorchVision_Interpret
"""

import torch
from PIL import Image
from torchvision.transforms import v2
from captum.attr import IntegratedGradients
import numpy as np

from src.model.vit import VisionTransformer, load_model
from src.dataset.dataset import load_dataset


import torch.nn.functional as F

def blur_baseline(x, kernel_size=31, sigma=5):
    # Simple Gaussian blur approximation
    x_blur = F.avg_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )
    return x_blur


if __name__ == '__main__':

    transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ])



    torch.manual_seed(123)
    np.random.seed(123)

    # Load Model
    params_path = "./runs/run8_cifar/params.json" # if you use load_model
    ckpt_path = "./runs/run8_cifar/epoch_100_model.pth"  # path to checkpoint used by load_model
    device = "cuda"

    # Load Model in Float32
    model, _ = load_model(params_path, ckpt_path, device)
    model.to(device)
    model.eval()
    model.zero_grad()

    # Image Proliferative DR
    target_class = 4
    image_path = r'/home/s-fx/fun/datasets/CIFAR-100-dataset/val/lion/king_of_beasts_s_001355.png'
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = torch.unsqueeze(image, 0).requires_grad_(True)
    image = image.to(device)
    #baseline = torch.zeros_like(image)
    baseline = blur_baseline(image)
    baseline = baseline.to(device)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(image, baseline, target=target_class, n_steps=50, internal_batch_size=5, return_convergence_delta=True)
    print('IG Attributions:', attributions)
    print('Convergence Delta:', delta)

    # Sum over channels
    attr = attributions.abs().sum(dim=1).squeeze(0)

    # Normalize to [0, 1]
    attr = attr - attr.min()
    attr = attr / (attr.max() + 1e-8)

    import matplotlib.pyplot as plt

    img = image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    heatmap = attr.detach().cpu().numpy()

    # Prepare image
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: Original image ----
    axes[0].imshow(img)
    axes[0].set_title("Original Fundus Image")
    axes[0].axis("off")

    # ---- Right: Integrated Gradients overlay ----
    axes[1].imshow(img)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title("Integrated Gradients Attribution")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


    k = 0.1  # top 10%
    threshold = torch.quantile(attr, 1 - k)

    masked = image.clone()
    masked[:, :, attr < threshold] = 0

    with torch.no_grad():
        original_score = model(image)[0, target_class]
        masked_score = model(masked)[0, target_class]

    print("Original:", original_score.item())
    print("Masked:", masked_score.item())

