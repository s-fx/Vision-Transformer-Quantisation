import sys
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
from src.dataset.augmentations import get_augmentations


class RetinopathyFullDataset(Dataset):
    def __init__(self, data_root, transform=None, mode='train'):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        if mode == 'train':
            self.images = glob.glob(f'{data_root}/train/*/*.jpg')
        elif mode == 'val':
            self.images = glob.glob(f'{data_root}/val/*/*.jpg')
        elif mode == 'test':
            self.images = glob.glob(f'{data_root}/test/*/*.jpg')
        elif mode == 'single':
           self.images = glob.glob(f'{data_root}/single_example/*/*.jpg')
        elif mode == 'eval':
            self.images = glob.glob(f'{data_root}/eval/*/*.jpg')
        else:
            print('Wrong mode for dataset creation.')
        #self.to_tensor = v2.ToTensor()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(image.split('/')[-2])
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return image, label



class RetinopathyAptosDataset(Dataset):
    def __init__(self, data_root, transform=None, train=True):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        if train:
            self.csv = pd.read_csv(f'{data_root}/train.csv')
            self.images = glob.glob(f'{data_root}/train_images/*.png')
        else:
            self.csv = pd.read_csv(f'{data_root}/test.csv')
            self.images = glob.glob(f'{data_root}/test_images/*.png')
        #self.to_tensor = v2.ToTensor()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_basename = os.path.basename(image).replace('.png','')
        image = Image.open(image).convert('RGB')
        diagnosis = self.csv.loc[self.csv['id_code'] == image_basename, 'diagnosis'].values[0]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return image, diagnosis



cifar100_classes = {
    'apple': 0,
    'aquarium_fish': 1,
    'baby': 2,
    'bear': 3,
    'beaver': 4,
    'bed': 5,
    'bee': 6,
    'beetle': 7,
    'bicycle': 8,
    'bottle': 9,
    'bowl': 10,
    'boy': 11,
    'bridge': 12,
    'bus': 13,
    'butterfly': 14,
    'camel': 15,
    'can': 16,
    'castle': 17,
    'caterpillar': 18,
    'cattle': 19,
    'chair': 20,
    'chimpanzee': 21,
    'clock': 22,
    'cloud': 23,
    'cockroach': 24,
    'couch': 25,
    'crab': 26,
    'crocodile': 27,
    'cup': 28,
    'dinosaur': 29,
    'dolphin': 30,
    'elephant': 31,
    'flatfish': 32,
    'forest': 33,
    'fox': 34,
    'girl': 35,
    'hamster': 36,
    'house': 37,
    'kangaroo': 38,
    'keyboard': 39,
    'lamp': 40,
    'lawn_mower': 41,
    'leopard': 42,
    'lion': 43,
    'lizard': 44,
    'lobster': 45,
    'man': 46,
    'maple_tree': 47,
    'motorcycle': 48,
    'mountain': 49,
    'mouse': 50,
    'mushroom': 51,
    'oak_tree': 52,
    'orange': 53,
    'orchid': 54,
    'otter': 55,
    'palm_tree': 56,
    'pear': 57,
    'pickup_truck': 58,
    'pine_tree': 59,
    'plain': 60,
    'plate': 61,
    'poppy': 62,
    'porcupine': 63,
    'possum': 64,
    'rabbit': 65,
    'raccoon': 66,
    'ray': 67,
    'road': 68,
    'rocket': 69,
    'rose': 70,
    'sea': 71,
    'seal': 72,
    'shark': 73,
    'shrew': 74,
    'skunk': 75,
    'skyscraper': 76,
    'snail': 77,
    'snake': 78,
    'spider': 79,
    'squirrel': 80,
    'streetcar': 81,
    'sunflower': 82,
    'sweet_pepper': 83,
    'table': 84,
    'tank': 85,
    'telephone': 86,
    'television': 87,
    'tiger': 88,
    'tractor': 89,
    'train': 90,
    'trout': 91,
    'tulip': 92,
    'turtle': 93,
    'wardrobe': 94,
    'whale': 95,
    'willow_tree': 96,
    'wolf': 97,
    'woman': 98,
    'worm': 99
}

cifar10_classes = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

class CIFAR100Dataset(Dataset):
    def __init__(self, data_root, transform=None, mode='train'):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        if mode == 'train':
            self.images = glob.glob(f'{data_root}/train/*/*.png')
        elif mode == 'val':
            self.images = glob.glob(f'{data_root}/val/*/*.png')
        elif mode == 'test':
            self.images = glob.glob(f'{data_root}/test/*/*.jpg')
        else:
            print('Wrong mode for dataset')
        #self.to_tensor = v2.ToTensor()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = str(image.split('/')[-2])
        label = int(cifar100_classes[label])
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        return image, label


class CIFAR10Dataset(Dataset):
    def __init__(self, data_root, transform=None, mode='train'):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        if mode == 'train':
            self.images = glob.glob(f'{data_root}/train/*/*.jpg')
        elif mode == 'val':
            self.images = glob.glob(f'{data_root}/val/*/*.jpg')
        elif mode == 'test':
            self.images = glob.glob(f'{data_root}/test/*/*.jpg')
        else:
            print('Wrong mode for dataset')
        #self.to_tensor = v2.ToTensor()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = str(image.split('/')[-2])
        label = int(cifar10_classes[label])
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        return image, label


def load_dataset(data_root):
    _, val_transform = get_augmentations(image_size=224)
    if 'retinopathy' in data_root:
        dataset_val = RetinopathyFullDataset(data_root, val_transform, mode='test')
        print('[*] Dataset: Retinopathy')
    elif 'CIFAR-100' in data_root:
        dataset_val = CIFAR100Dataset(data_root, val_transform, mode='test')
        print('[*] Dataset: CIFAR100')
    elif 'CIFAR-10' in data_root:
        dataset_val = CIFAR10Dataset(data_root, val_transform, mode='test')
        print('[*] Dataset: CIFAR100')
    else:
        print('No Dataset given. Exit')
        sys.exit()

    test_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    return test_dataloader


if __name__ == '__main__':
    #dataset = RetinopathyFullDataset('/home/s-fx/fun/datasets/retinopathy-full-ds', transform=None, mode='test')
    dataset = CIFAR100Dataset('/home/s-fx/fun/datasets/CIFAR-100-dataset', transform=None, mode='val')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataset)
    print(data_loader)
    for idx, (img, label) in enumerate(data_loader):
        print(img)
        print(label)
