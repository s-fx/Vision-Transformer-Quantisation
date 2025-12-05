import torch
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

