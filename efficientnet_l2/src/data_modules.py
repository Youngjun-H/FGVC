# data_modules.py (수정 예시)

import lightning as L
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np

class ImageFolderDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int, num_workers: int = 4, 
                 train_indices=None, val_indices=None):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        # K-Fold를 위한 인덱스 저장
        self.train_indices = train_indices
        self.val_indices = val_indices

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 검증 및 테스트 데이터용 변환 (보통 Augmentation 없음)
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 클래스 정보를 저장하기 위해 setup에서 초기화
        self.num_classes = 0
        self.class_to_idx = {}

    def setup(self, stage: str = None):
        # 전체 데이터셋을 한 번만 로드
        full_dataset = ImageFolder(self.data_dir)
        self.num_classes = len(full_dataset.classes)
        self.class_to_idx = full_dataset.class_to_idx

        # train/val 인덱스가 제공된 경우 Subset을 사용
        if self.train_indices is not None and self.val_indices is not None:
            # 훈련 데이터셋에는 증강(augmentation) 적용
            self.train_dataset = Subset(ImageFolder(self.data_dir, transform=self.transform), self.train_indices)
            # 검증 데이터셋에는 증강 미적용
            self.val_dataset = Subset(ImageFolder(self.data_dir, transform=self.val_transform), self.val_indices)
        else:
            # K-Fold가 아닐 경우, 기존 방식 (예: 랜덤 분할) 사용 (옵션)
            # 여기서는 K-Fold 전용으로 가정하고 단순화합니다.
            raise ValueError("train_indices and val_indices must be provided for K-Fold training.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)