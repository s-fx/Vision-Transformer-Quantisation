import torch
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.dino_vit import get_transforms
from src.dataset.dataset import RetinopathyFullDataset
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from src.quant.utils import benchmark


def accuracy_fn(y_pred, y_true):
    """
    y_pred: (batch_size, num_classes) raw logits
    y_true: (batch_size,) ground truth class indices
    """
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).sum().item()
    acc = correct / y_true.size(0)
    return acc


def plot_loss(loss_dict, out_dir):
    if isinstance(loss_dict, str):
        with open(loss_dict, 'rb') as f:
            loss_dict = pickle.load(f)

    train_loss_epoch = loss_dict['train_loss_epoch']
    valid_loss_epoch = loss_dict['valid_loss_epoch']

    fig, ax = plt.subplots()
    ax.plot(train_loss_epoch, color='blue', label='Train Loss')
    ax.plot(valid_loss_epoch, color='red', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training / Validation Loss')
    ax.legend(loc='upper right')
    fig.savefig(f'{out_dir}/loss_epoch.png')
    plt.close()


def animate_weight_distr(snapshots_path):
    snapshots = torch.load(snapshots_path, weights_only=False)

    layer_names = list(snapshots.keys())
    num_layers = len(layer_names)
    epochs = len(next(iter(snapshots.values())))

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4), sharey=True)

    def update(frame_idx):
        for ax, name in zip(axes, layer_names):
            ax.clear()
            data = snapshots[name][frame_idx]
            sns.histplot(data, bins=80, kde=True, ax=ax, color='royalblue')
            ax.set_xlim(-0.1, 0.1)
            ax.set_ylim(0, 20000)
            ax.set_title(f"{name}\nEpoch {frame_idx * 5}")
            ax.set_xlabel("Weight Value")
            mean, std = data.mean(), data.std()
            ax.text(0.05, 0.95, f"μ={mean:.4f}\nσ={std:.4f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment='top')

    fig.suptitle("ViT Weight Distribution Evolution Across Layers", fontsize=14)
    anim = FuncAnimation(fig, update, frames=epochs, interval=150)
    anim.save("vit_layer_comparison.gif", writer=PillowWriter(fps=10))
    plt.close(fig)

    print("[*] Saved side-by-side animation as vit_layer_comparison.gif")


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


def plot_metrics(acc, prec, rec, f1, mode):
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
    plt.savefig(f'./metrics_{mode}.png')


def plot_cm(cm, class_names, mode):
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
    plt.title(f"Confusion Matrix {mode}", fontsize=16)

    plt.xticks(rotation=45, ha="right", fontsize=12)  # rotate x-labels
    plt.yticks(rotation=0, fontsize=12)               # keep y-labels horizontal

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{mode}.png')


def run_evaluation_retino(model, model_name, root, data_root, params_path, ckpt_path, loss_dict, mode):
    print(mode)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    #weight_snapshot = os.path.join(root, 'vit_weight_snapshots_100.pt')

    # Plot Loss
    #plot_loss(loss_dict, './runs/')
    # Animate Weight Distribution
    #animate_weight_distr(weight_snapshot)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'[INFO] Device in use: {device}')
    print(model)

    if model_name == 'dino':
        _, val_transforms = get_transforms()
        dataset_val = RetinopathyFullDataset(data_root, val_transforms, mode='eval')
        test_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    else:
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

    print(f'Test Dataset: {len(test_dataloader)}')
    start_time = time.time()
    for idx, (image, label) in enumerate(tqdm(test_dataloader)):

        image = image.to(device)
        label = label.to(device)

        pred_logits = model(image)

        preds = torch.argmax(pred_logits, dim=1)
        label = label.cpu().numpy().item()
        preds = preds.cpu().numpy().item()
        labels_distr[str(label)] += 1
        all_preds.append(preds)
        all_labels.append(label)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed {elapsed_time}')

    plot_classes(labels_distr, class_names)

    cm = confusion_matrix(all_labels, all_preds)
    np.save(f'confusion_matrix_{mode}.npy', cm)
    print(cm)

    # Plot Confusion Matrix
    plot_cm(cm, class_names, mode)

    # Calculate Accuracy, Precision, Recall, F1
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    plot_metrics(acc, prec, rec, f1, mode)

    print(f"Accuracy {mode}:", acc)
    print(f"Precision {mode}:", prec)
    print(f"Recall {mode}:", rec)
    print(f"F1 {mode}:", f1)

    # Run benchmark


    txt_filename = f'./results_{mode}.txt'
    with open(txt_filename, 'w') as f:
        f.writelines(f'Total time {elapsed_time}\n')
        f.writelines(f'Test Dataset {len(test_dataloader)}\n')
        f.writelines(f'Accuracy {acc}\n')
        f.writelines(f'Precision {prec}\n')
        f.writelines(f'Recall {rec}\n')
        f.writelines(f'F1 {f1}\n')
        f.writelines(f'Len Dataset {len(test_dataloader)}\n')
