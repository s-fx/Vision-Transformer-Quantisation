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

from src.dataset.dataset import RetinopathyAptosDataset, RetinopathyFullDataset, CIFAR100Dataset
from src.model.vit import VisionTransformer
from src.eval import plot_loss, animate_weight_distr
from src.dataset.augmentations import get_augmentations


def load_model(params_path, ckpt_path, device):
    with open(params_path, 'r') as f:
        p = json.load(f)

    # Load Model
    model = VisionTransformer(in_channels=p['in_channels'],
                              image_size=p['image_size'],
                              patch_size=p['patch_size'],
                              number_of_encoder=p['number_of_encoder'],
                              embeddings=p['embeddings'],
                              d_ff_scale=p['d_ff_scale'],
                              num_heads=p['num_heads'],
                              input_dropout_rate=p['input_dropout_rate'],
                              attention_dropout_rate=p['attention_dropout_rate'],
                              feed_forward_dropout_rate=p['feed_forward_dropout_rate'],
                              number_of_classes=p['number_of_classes']
                             )

    # Load weights and loss dict
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_dict = checkpoint.get('loss_dict', {})

    model.to(device)
    model.eval()
    return model, loss_dict


def load_dataset(data_root):
    _, val_transform = get_augmentations(image_size=224)
    if 'retinopathy' in data_root:
        dataset_val = RetinopathyFullDataset(data_root, val_transform, mode='test')
        print('[*] Dataset: Retinopathy')
    elif 'cifar' in data_root:
        dataset_val = CIFAR100Dataset(data_root, train_transform, mode='val')
        print('[*] Dataset: CIFAR100')
    else:
        print('No Dataset given. Exit')
        sys.exit()

    test_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return test_dataloader


def plot_cm(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 12})  # larger numbers inside cells

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)

    plt.xticks(rotation=45, ha="right", fontsize=12)  # rotate x-labels
    plt.yticks(rotation=0, fontsize=12)               # keep y-labels horizontal

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


def plot_metrics(acc, prec, rec, f1):
    scores = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1
    }

    plt.figure(figsize=(6,4))
    plt.bar(scores.keys(), scores.values())
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Overall Evaluation Metrics")
    plt.savefig('./metrics.png')


def plot_classes(labels_distr, class_names):
    keys = [int(k) for k in labels_distr.keys()]
    counts = [labels_distr[k] for k in labels_distr.keys()]

    # Map class numbers → class names
    classes = [class_names[k] for k in keys]

    plt.figure(figsize=(10,5))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Image Distribution per Class")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('class_distribution.png')


if __name__ == '__main__':
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    root = './runs/run7_retino_224/'
    data_root = '/home/s-fx/fun/datasets/retinopathy-full-ds'
    params_path = os.path.join(root, 'params.json')
    ckpt_path = os.path.join(root, 'epoch_100_model.pth')
    loss_dict = os.path.join(root, 'loss_dict.pkl')
    weight_snapshot = os.path.join(root, 'vit_weight_snapshots_100.pt')

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
        labels_distr[str(label)] += 1
        all_preds.append(preds)
        all_labels.append(label)

    plot_classes(labels_distr, class_names)

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
