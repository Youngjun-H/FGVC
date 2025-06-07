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
    # ⭐️ 생성자를 매우 단순하게 변경
    def __init__(self, model_variant: str, num_classes: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        
        model_loader, weights, _ = MODEL_MAP[model_variant]
        self.model = model_loader(weights=weights)

        # Transfer Learning: 분류층만 새로 정의하고 이 부분만 학습
        in_features = self.model.classifier[1].in_features
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 이제 분류층 파라미터만 학습합니다.
        return optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate)