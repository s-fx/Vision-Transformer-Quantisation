import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2


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
        else:
            print('Wrong mode for dataset creation.')
        self.to_tensor = v2.ToTensor()

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
        self.to_tensor = v2.ToTensor()


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
        else:
            print('Wrong mode for dataset')
        self.to_tensor = v2.ToTensor()

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



if __name__ == '__main__':
    #dataset = RetinopathyFullDataset('/home/s-fx/fun/datasets/retinopathy-full-ds', transform=None, mode='test')
    dataset = CIFAR100Dataset('/home/s-fx/fun/datasets/CIFAR-100-dataset', transform=None, mode='val')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataset)
    print(data_loader)
    for idx, (img, label) in enumerate(data_loader):
        print(img)
        print(label)
