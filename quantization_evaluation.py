import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob

from src.model.vit import load_model
from src.dino_vit import load_dino, visualise_features, get_transforms
from src.quant.quantLayer_sym import replace_linear_layers, QuantizedLinearLayer
#from src.quant.quantLayer_asym import replace_linear_layers, QuantizedLinearLayer
from src.eval import run_evaluation_retino
from src.quant.utils import get_model_size, calc_model_size_mb, get_model_size_bytes, benchmark


def main():
    root = './runs/run_dino_retino_no_backbone'
    data_root = '/home/s-fx/fun/datasets/retinopathy-full-ds-cleaned'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_32_model.pth')
    loss_dict = os.path.join(root, 'loss_dict.pkl')
    device = 'cuda'
    images_path = glob.glob(f'{data_root}/single_example/*/*.jpg')
    _, val_transforms = get_transforms()

    dummy_image = Image.open(f'{data_root}/single_example/0/0d744aed4d64-600.jpg')
    dummy_image = val_transforms(dummy_image).unsqueeze(0).to(device)

    # ------------ BASE MODEL --------------
    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    #model = torch.compile(model, mode='max-autotune')
    print(model)
    get_model_size(model)
    get_model_size_bytes(model)
    model.to(device)
    res = benchmark(model, dummy_image)
    print(res)
    #model.eval()

    # Visualise Features
    for img_path in images_path:
        visualise_features(model, img_path, device)


    # Run Evaluation on Base Model (Backbone trained)
    run_evaluation_retino(model, 'dino', root, data_root, params_path, ckpt_path, loss_dict, mode='base')


    # ----------- QUANTISATION SYMETRIC-----------
    print(30*'-')

    print(f'\nQuantization')
    model = load_dino(params_path, ckpt_path, device)
    model.eval()
    replace_linear_layers(model, QuantizedLinearLayer, [''], quantized=True)
    #model = torch.compile(model, mode='max-autotune')
    print(model)
    get_model_size(model)
    get_model_size_bytes(model)

    model.to(device)
    res = benchmark(model, dummy_image)
    print(res)

    # Visualise Features
    for img_path in images_path:
        visualise_features(model, img_path, device)

    # Run Evaluation on Quantised Model (Backbone trained)
    run_evaluation_retino(model, 'dino', root, data_root, params_path, ckpt_path, loss_dict, mode='quant_asym')


if __name__ == '__main__':

    main()
