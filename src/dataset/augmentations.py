from torchvision.transforms import v2

def get_augmentations(image_size):
    train_transform = v2.Compose([
        v2.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # encourages scale invariance
        v2.TrivialAugmentWide(),                                                # mild color & contrast jitter
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),                                           # retinal symmetry helps
        v2.RandomRotation(degrees=15),                                          # small rotations only
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
    v2.Resize((image_size, image_size)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

