import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import SEED, IMG_SIZE, VAL_SPLIT, BATCH_SIZE, DATA_TRAIN, DATA_TEST, MEAN, STD


def get_transforms(augment: bool = False) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def build_loaders(batch_size=BATCH_SIZE, augment=True):
    full_train = datasets.ImageFolder(DATA_TRAIN, transform=get_transforms(augment))
    n_val = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # validation always uses clean transforms, even when training uses augmentation
    val_ds.dataset = datasets.ImageFolder(DATA_TRAIN, transform=get_transforms(False))
    test_ds = datasets.ImageFolder(DATA_TEST, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, full_train.classes


def make_grid_loader(batch_size: int) -> DataLoader:
    ds = datasets.ImageFolder(DATA_TRAIN, transform=get_transforms(True))
    n_val = int(len(ds) * VAL_SPLIT)
    n_train = len(ds) - n_val
    train_ds, _ = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    return DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, drop_last=True, num_workers=0)
