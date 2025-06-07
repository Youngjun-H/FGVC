import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import lightning as L
from config import IMG_SIZE, VALIDATION_SPLIT

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class EfficientNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # Load full dataset
        full_dataset = ImageFolder(self.data_dir)
        
        # Split dataset
        num_train = len(full_dataset)
        val_size = int(VALIDATION_SPLIT * num_train)
        train_size = num_train - val_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
        
        # Apply transforms
        self.train_dataset = TransformedDataset(train_subset, transform=self.train_transforms)
        self.val_dataset = TransformedDataset(val_subset, transform=self.val_transforms)
        
        self.class_names = full_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 