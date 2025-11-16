import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn as nn

from src.dataset.dataset import RetinopathyAptosDataset, RetinopathyFullDataset, CIFAR100Dataset
from src.dataset.augmentations import get_augmentations
from src.model.vit import VisionTransformer
from src.eval import accuracy_fn

from tqdm import tqdm
import pickle
import json
import glob
import os
import math


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

    # Optional: reload the loss_dict.pkl file if you want the full history
    #loss_dict_path = 'runs/loss_dict.pkl'
    #if os.path.exists(loss_dict_path):
    #    with open(loss_dict_path, 'rb') as file:
    #        loss_dict = pickle.load(file)
    return model, optimizer, scheduler, start_epoch, loss_dict


def main(p):
    data_root = p['data_root']
    num_epochs = p['num_epochs']

    # Create transformations and load Dataset
    train_transform, val_transform = get_augmentations(p['image_size'])
    print(f'[==>] Using augmentations {train_transform}\n{val_transform}')

    if 'retinopathy' in data_root:
        dataset_train = RetinopathyFullDataset(data_root, train_transform, mode='train')
        dataset_val = RetinopathyFullDataset(data_root, val_transform, mode='val')
        print('[*] Dataset: Retinopathy')
    elif 'cifar' in data_root:
        dataset_train = CIFAR100Dataset(data_root, train_transform, mode='train')
        dataset_val = CIFAR100Dataset(data_root, train_transform, mode='val')
        print('[*] Dataset: CIFAR100')
    else:
        print('No Dataset given. Exit')
        sys.exit()

    train_dataloader = DataLoader(dataset_train, batch_size=p['batch_size'], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=p['batch_size'], shuffle=False)

    # Create Model
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
        model.parameters(),
        lr=p['lr'],
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

    with open(f'runs/loss_dict.pkl', 'wb') as file:
        pickle.dump(loss_dict, file)

    print('[++] Done.')



if __name__ == '__main__':
    params_dict = {
        # Data
        'num_epochs': 100,
        'data_root': '/home/s-fx/fun/datasets/retinopathy-full-ds',
        'batch_size': 256,
        'resume': False,
        # ViT
        'in_channels': 3,
        'image_size': 224,
        'patch_size': 16,
        'number_of_encoder': 6,
        'embeddings': 512,
        'd_ff_scale': 4,
        'num_heads': 8,
        'input_dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'feed_forward_dropout_rate': 0.1,
        'number_of_classes': 5,
        # Optimizer
        'lr': 0.001,
        'betas': (0.9,0.999),
        'eps': 1e-8,
        'weight_decay': 0.1
    }

    with open("runs/params.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    main(p=params_dict)

