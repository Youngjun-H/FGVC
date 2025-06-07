import lightning as L
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
from torchmetrics import Accuracy

# 모델 선택 로직을 딕셔너리로 관리하여 가독성 향상
MODEL_MAP = {
    'B0': (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, (224, 224)),
    'B1': (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1, (240, 240)),
    'B2': (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, (260, 260)),
    'B3': (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, (300, 300)),
}

class EfficientNetFineTuner(L.LightningModule):
    # ⭐️ 아래 줄의 __init__ 메서드 정의를 수정합니다.
    def __init__(self, model_variant: str, num_classes: int, phase1_epochs: int, phase2_epochs: int, learning_rate_phase1: float, learning_rate_phase2: float):
        super().__init__()
        # 하이퍼파라미터를 저장합니다.
        self.save_hyperparameters()
        
        if model_variant not in MODEL_MAP:
            raise ValueError("Unsupported EfficientNet variant")
        
        model_loader, weights, _ = MODEL_MAP[model_variant]

        # 모델 구축 (이하 변경 없음)
        self.model = model_loader(weights=weights)

        # 1. 모든 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 분류층 교체 및 동결 해제
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 손실 함수 및 평가지표 정의
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    # ... (forward, on_train_epoch_start, training_step 등 다른 메서드는 변경 없음)
    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.phase1_epochs:
            print(f"\n--- Epoch {self.current_epoch}: Switching to Fine-Tuning Phase ---")
            for param in self.model.parameters():
                param.requires_grad = True
            
            optimizer = self.optimizers()
            optimizer.param_groups[0]['lr'] = self.hparams.learning_rate_phase2
            print(f"Set learning rate to {self.hparams.learning_rate_phase2}")

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(logits, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate_phase1)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }