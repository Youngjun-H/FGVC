import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# random_split 후 각 데이터셋에 맞는 transform을 적용하기 위한 헬퍼 클래스
class TransformedDataset(torch.utils.data.Dataset):
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

class ImageFolderDataModule(L.LightningDataModule):
    def __init__(self, train_dir: str, val_split: float, batch_size: int, num_workers: int, image_size: tuple = (224, 224)):
        super().__init__()
        self.train_dir = train_dir
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # 데이터 증강 및 정규화 정의
        self.train_transforms = T.Compose([
            T.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0)),
            T.RandomRotation(degrees=20),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transforms = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage: str):
        # ImageFolder는 클래스별 폴더 구조를 가정합니다.
        full_dataset = ImageFolder(self.train_dir)
        num_train = len(full_dataset)
        val_size = int(self.val_split * num_train)
        train_size = num_train - val_size
        
        # 데이터셋 분할
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # 각 분할에 맞는 transform 적용
        if stage == "fit" or stage is None:
            self.train_dataset = TransformedDataset(train_subset, transform=self.train_transforms)
            self.val_dataset = TransformedDataset(val_subset, transform=self.val_transforms)

        # test stage가 필요하다면 여기에 로직 추가
        # if stage == "test" or stage is None:
        #     self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)