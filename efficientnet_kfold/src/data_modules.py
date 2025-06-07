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

class ImageFolderDataModule(L.LightningModule):
    # ⭐️ train_indices, val_indices를 인자로 받도록 수정
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, image_size: tuple, train_indices=None, val_indices=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # Transform 정의 (이전과 동일)
        self.train_transforms = T.Compose([...])
        self.val_transforms = T.Compose([...])

    def setup(self, stage: str):
        # ⭐️ K-Fold 인덱스를 받아 Subset을 생성하는 로직으로 변경
        full_dataset = ImageFolder(self.data_dir)
        
        # Subset에 각각 다른 transform을 적용하기 위해 별도 클래스 사용
        self.train_dataset = TransformedSubset(full_dataset, self.train_indices, self.train_transforms)
        self.val_dataset = TransformedSubset(full_dataset, self.val_indices, self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Subset에 transform을 적용하기 위한 헬퍼 클래스
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label
    def __len__(self):
        return len(self.indices)