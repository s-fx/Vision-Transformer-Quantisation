import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from src.dataset.dataset import RetinopathyAptosDataset, RetinopathyFullDataset, CIFAR100Dataset, load_dataset, cifar100_classes
from src.model.vit import VisionTransformer, load_model
from src.eval import plot_loss, animate_weight_distr, plot_classes, plot_metrics, plot_cm
from src.dataset.augmentations import get_augmentations





def main():

    root = './runs/run6_cifar_224'
    data_root = '/home/s-fx/fun/datasets/CIFAR-100-dataset'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_100_model.pth')
    loss_dict = os.path.join(root, 'loss_dict.pkl')
    #weight_snapshot = os.path.join(root, 'vit_weight_snapshots_100.pt')
    class_names = cifar100_classes

    # Plot Loss
    #plot_loss(loss_dict, './runs/')
    # Animate Weight Distribution
    #animate_weight_distr(weight_snapshot)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'[INFO] Device in use: {device}')
    model, loss_dict = load_model(params_path, ckpt_path, device)
    print(model)
    print(loss_dict.keys())
    test_dataloader = load_dataset(data_root)

    all_preds = []
    all_labels = []
    labels_distr = {
        '0': 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0
    }

    for idx, (image, label) in enumerate(test_dataloader):
        print(f'Working on {idx}')

        image = image.to(device)
        label = label.to(device)

        pred_logits = model(image)

        preds = torch.argmax(pred_logits, dim=1)
        label = label.cpu().numpy().item()
        preds = preds.cpu().numpy().item()
        print(f'Prediction: {preds}\nGround Truth: {label}')
        #labels_distr[str(label)] += 1
        all_preds.append(preds)
        all_labels.append(label)

    #plot_classes(labels_distr, class_names)

    cm = confusion_matrix(all_labels, all_preds)
    np.save('confusion_matrix.npy', cm)
    print(cm)

    # Plot Confusion Matrix
    plot_cm(cm, class_names)

    # Calculate Accuracy, Precision, Recall, F1
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    plot_metrics(acc, prec, rec, f1)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)


if __name__ == '__main__':
    main()
