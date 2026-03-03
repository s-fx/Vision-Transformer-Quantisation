import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn as nn

from src.dataset.dataset import RetinopathyAptosDataset, RetinopathyFullDataset, CIFAR100Dataset, CIFAR10Dataset
from src.dataset.augmentations import get_augmentations
from src.model.vit import VisionTransformer
from src.eval import accuracy_fn, plot_loss
from src.utils import check_loaded_layers, load_pretrained_vit_weights, compare_state_dicts, compare_weight_distributions, \
        reinit_classification_head
from src.dino_vit import DinoVisionTransformerClassifier, get_transforms

from tqdm import tqdm
import pickle
import json
import glob
import os
import math
import sys
from collections import defaultdict, Counter


def cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def resume_training(p, model, optimizer, scheduler, device):
    ckpts = glob.glob('runs/epoch_*_model.pth')
    if ckpts:
        ckpt_path = max(ckpts, key=os.path.getctime)

    start_epoch = 0
    loss_dict = {}

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_dict = checkpoint.get('loss_dict', {})

        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting fresh training.")

    return model, optimizer, scheduler, start_epoch, loss_dict


def main(p):
    data_root = p['data_root']
    num_epochs = p['num_epochs']

    # Create transformations and load Dataset
    train_transform, val_transform = get_transforms(image_dimension=256)

    if 'retinopathy' in data_root.lower():
        dataset_train = RetinopathyFullDataset(data_root, train_transform, mode='train')
        dataset_val = RetinopathyFullDataset(data_root, val_transform, mode='val')
        print('[*] Dataset: Retinopathy')
    elif 'cifar-100' in data_root.lower():
        dataset_train = CIFAR100Dataset(data_root, train_transform, mode='train')
        dataset_val = CIFAR100Dataset(data_root, train_transform, mode='val')
        print('[*] Dataset: CIFAR100')
    elif 'cifar-10' in data_root.lower():
        dataset_train = CIFAR10Dataset(data_root, train_transform, mode='train')
        dataset_val = CIFAR10Dataset(data_root, train_transform, mode='val')
        print('[*] Dataset: CIFAR10')
    else:
        print('No Dataset given. Exit')
        sys.exit()

    if 'retinopathy' in data_root.lower():
        # Balanced sampling per epoch (counter class imbalance)
        labels = [int(p.split('/')[-2]) for p in dataset_train.images]
        class_counts = Counter(labels)
        print(f'Class counts: {class_counts}')
        class_weights = {
            cls: 1.0 / count
            for cls, count in class_counts.items()
        }
        sample_weights = torch.tensor(
            [class_weights[label] for label in labels],
            dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_dataloader = DataLoader(dataset_train, batch_size=p['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(dataset_val, batch_size=p['batch_size'], shuffle=False)
    else:
        train_dataloader = DataLoader(dataset_train, batch_size=p['batch_size'], shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=p['batch_size'], shuffle=False)


    # Create Model
    model = DinoVisionTransformerClassifier(n_classes=p['number_of_classes'])

    # Train all layers
    for param in model.parameters():
        param.requires_grad = True

    # Freeze backbone if param is set to True
    if p['freeze_backbone']:
        for param in model.transformer.parameters():
            param.requires_grad = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'[INFO] Device in use: {device}')
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Number of trainable parameters: {trainable_params}')

    # Optimizer
    optimizer = optim.AdamW(
        [
            {
                'params': model.transformer.parameters(),
                'lr': p['lr_backbone']
            },
            {
                'params': model.classifier.parameters(),
                'lr': p['lr']
            }
        ],
        betas=p['betas'],
        eps=p['eps'],
        weight_decay=p['weight_decay']
    )
    num_epochs = p['num_epochs']
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warm-up

    scheduler = cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    loss_dict = {
        'train_loss': [], 'valid_loss': [],
        'train_loss_epoch': [], 'valid_loss_epoch': [],
        'train_acc': [], 'valid_acc': [],
        'train_acc_epoch': [], 'valid_acc_epoch': []
    }

    train_loss = []
    train_acc = []
    n_train = len(train_dataloader)
    start_epoch = 0

    if p['resume']:
        model, optimizer, scheduler, start_epoch, loss_dict = resume_training(p, model, optimizer, scheduler, device)
        model.to(device)


    for epoch in range(start_epoch, num_epochs):
        print(f'[TRAINING] Epoch {epoch+1}')
        tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train()
        loss_average = 0
        accuracy_average = 0
        for idx, (img, label) in enumerate(tqdm_bar):

            optimizer.zero_grad()

            img = img.to(device)
            label = label.to(device)
            pred_logits = model(img)
            loss = criterion(pred_logits, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            loss_average += loss.item()
            accuracy = accuracy_fn(pred_logits, label)
            accuracy_average += accuracy_fn(pred_logits, label)
            current_lr = optimizer.param_groups[0]['lr']
            tqdm_bar.set_description(desc=f'Training Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | LR: {current_lr:.8f}')
            loss_dict['train_loss'].append(loss_average)
            loss_dict['train_acc'].append(accuracy_average)

        epoch_avg_loss = loss_average / n_train
        epoch_avg_accuracy = accuracy_average / n_train
        train_loss.append(epoch_avg_loss)
        train_acc.append(epoch_avg_accuracy)
        loss_dict['train_loss_epoch'].append(epoch_avg_loss)
        loss_dict['train_acc_epoch'].append(epoch_avg_accuracy)
        print(f'Epoch {epoch+1} | Loss {epoch_avg_loss:.4f} | Accuracy {epoch_avg_accuracy:.4f}')


        print(f'[VALIDATION] Epoch {epoch+1}')
        tqdm_bar_val = tqdm(val_dataloader, total=len(val_dataloader))
        val_loss_list = []
        model.eval()
        val_avg_loss = 0
        val_avg_acc = 0
        n_val = len(val_dataloader)

        with torch.no_grad():
            for idx, (img, label) in enumerate(tqdm_bar_val):
                img = img.to(device)
                label = label.to(device)
                pred_logits = model(img)
                loss = criterion(pred_logits, label)
                val_avg_loss += loss.item()
                val_acc = accuracy_fn(pred_logits, label)
                val_avg_acc += accuracy_fn(pred_logits, label)
                loss_dict['valid_loss'].append(val_avg_loss)
                loss_dict['valid_acc'].append(val_avg_acc)
            tqdm_bar_val.set_description(desc=f'Validation Loss: {loss:.4f} | Accuracy: {val_acc:.4f}')
            val_epoch_avg_loss = val_avg_loss / n_val
            val_epoch_avg_acc = val_avg_acc / n_val
            loss_dict['valid_loss_epoch'].append(val_epoch_avg_loss)
            loss_dict['valid_acc_epoch'].append(val_epoch_avg_acc)
            print(f'Epoch {epoch+1} | Loss {val_epoch_avg_loss:.4f} | Accuracy {val_epoch_avg_acc:.4f}')


        # Save model checkpoint
        if epoch % 1 == 0:
            ckpt_file_name = f'runs/epoch_{epoch+1}_model.pth'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_dict': loss_dict
            }, ckpt_file_name)

        # Plot Loss
        plot_loss(loss_dict, './runs/')

    with open(f'runs/loss_dict.pkl', 'wb') as file:
        pickle.dump(loss_dict, file)

    print('[++] Done.')



if __name__ == '__main__':
    params_dict = {
        # Data
        'num_epochs': 100,
        'data_root': '/home/s-fx/fun/datasets/retinopathy-full-ds-cleaned',
        'batch_size': 64,
        'resume': False,
        # ViT
        'number_of_classes': 5,
        'freeze_backbone': False,
        # Optimizer
        'lr': 1e-3,
        'lr_backbone': 1e-5,
        'betas': (0.9,0.999),
        'eps': 1e-8,
        'weight_decay': 0.01
    }

    with open("runs/params.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    main(p=params_dict)

