import lightning as L
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_V2_L_Weights
from torchmetrics import Accuracy

# 모델 선택 로직 (변경 없음)
MODEL_MAP = {
    'B0': (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, (224, 224)),
    'B1': (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1, (240, 240)),
    'B2': (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, (260, 260)),
    'B3': (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, (300, 300)),
    'V2L': (models.efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1, (384, 384)),
}

class EfficientNetFineTuner(L.LightningModule):
    # __init__ 메서드는 변경 없음
    def __init__(self, model_variant: str, num_classes: int, phase1_epochs: int, phase2_epochs: int, learning_rate_phase1: float, learning_rate_phase2: float):
        super().__init__()
        self.save_hyperparameters()
        
        if model_variant not in MODEL_MAP:
            raise ValueError("Unsupported EfficientNet variant")
        
        model_loader, weights, _ = MODEL_MAP[model_variant]
        self.model = model_loader(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False
        
        # EfficientNet의 특징(features) 레이어와 분류기(classifier)를 명확히 구분
        self.features = self.model.features
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        self.classifier = self.model.classifier

        for param in self.classifier.parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

    # ✨ [수정 1] 차등 학습률을 적용하도록 on_train_epoch_start 수정
    def on_train_epoch_start(self):
        """
        Phase 2 시작 시, 백본 전체의 동결을 풀고 차등 학습률을 적용합니다.
        """
        if self.current_epoch == self.hparams.phase1_epochs:
            print(f"\n--- Epoch {self.current_epoch}: Switching to Fine-Tuning Phase (Rank {self.global_rank}) ---")
            
            # 1. 모델 전체 파라미터 동결 해제
            for param in self.parameters():
                param.requires_grad = True
            
            # 2. Optimizer에 백본 파라미터 그룹을 '아주 낮은' 학습률로 추가
            optimizer = self.optimizers()
            
            # 백본(features) 파라미터를 위한 새로운 파라미터 그룹 추가
            # phase2 학습률은 백본에 적용 (아주 작은 값 권장, 예: 1e-5)
            optimizer.add_param_group({
                'params': self.features.parameters(),
                'lr': self.hparams.learning_rate_phase2 
            })

            # 기존 분류기(classifier) 그룹의 학습률은 phase1 값으로 유지하거나 재설정할 수 있습니다.
            # 여기서는 phase1 학습률을 그대로 사용하도록 설정합니다.
            classifier_lr = self.hparams.learning_rate_phase1
            optimizer.param_groups[0]['lr'] = classifier_lr

            print(f"Optimizer reconfigured for Phase 2 (Rank {self.global_rank}):")
            print(f"  - Classifier LR: {classifier_lr}")
            print(f"  - Backbone LR:   {self.hparams.learning_rate_phase2}")

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(logits, labels)
        
        # 현재 학습률을 로그로 남겨서 확인
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        acc = self.accuracy(logits, labels)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    # ✨ [수정 2] AdamW 사용 및 Phase 1 옵티마이저 설정 명확화
    def configure_optimizers(self):
        """
        Phase 1에서는 분류기(classifier)의 파라미터만 학습하도록 옵티마이저를 설정합니다.
        """
        # Phase 1: 분류기만 학습
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.hparams.learning_rate_phase1)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }