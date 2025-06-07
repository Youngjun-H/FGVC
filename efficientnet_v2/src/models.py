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
        logits = self.model(x)
        return logits
    
    # # ⭐️ 3. 최적의 온도를 찾는 메서드 추가
    # def find_optimal_temperature(self, val_loader: torch.utils.data.DataLoader):
    #     """
    #     검증 데이터셋을 사용하여 최적의 온도를 찾고, self.temperature를 업데이트합니다.
    #     """
    #     self.eval() # 모델을 평가 모드로 설정
        
    #     all_logits = []
    #     all_labels = []

    #     # 먼저 검증 데이터셋 전체에 대한 logits와 레이블을 수집합니다.
    #     print("Collecting logits from validation set for calibration...")
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs = inputs.to(self.device)
                
    #             # 원본 모델의 순수 logits를 얻기 위해 self.model을 직접 호출
    #             logits = self.model(inputs) 
    #             all_logits.append(logits)
    #             all_labels.append(labels)
        
    #     all_logits = torch.cat(all_logits).to(self.device)
    #     all_labels = torch.cat(all_labels).to(self.device)

    #     # 최적의 온도를 찾기 위한 최적화 시작
    #     # 온도를 학습 가능한 파라미터로 설정 (초기값 1.5)
    #     temperature_param = nn.Parameter(torch.ones(1).to(self.device) * 1.5)
        
    #     # 손실 함수 (NLL Loss, CrossEntropyLoss와 동일)
    #     nll_criterion = nn.CrossEntropyLoss()

    #     # L-BFGS 옵티마이저는 이런 단일 변수 최적화에 효과적입니다.
    #     optimizer = optim.LBFGS([temperature_param], lr=0.01, max_iter=50)

    #     def closure():
    #         optimizer.zero_grad()
    #         # 온도로 스케일링된 logits에 대한 손실 계산
    #         loss = nll_criterion(all_logits / temperature_param, all_labels)
    #         loss.backward()
    #         return loss
        
    #     print("Finding optimal temperature...")
    #     optimizer.step(closure)

    #     # optimal_t = temperature_param.item()
    #     # print(f"Optimal temperature found: {optimal_t:.4f}")
        
    #     # # 찾은 최적의 온도를 모델의 버퍼에 저장
    #     # self.temperature.data = torch.tensor(optimal_t)

    #     return temperature_param.item()
    
    # # # ⭐️ 2. 보정 전/후 성능을 체계적으로 검증하는 메서드 추가
    # # def test_calibration(self, val_loader: torch.utils.data.DataLoader):
    # #     self.eval()
        
    # #     all_logits = []
    # #     all_labels = []

    # #     print("Collecting data for calibration test...")
    # #     with torch.no_grad():
    # #         for inputs, labels in val_loader:
    # #             inputs = inputs.to(self.device)
    # #             logits = self.model(inputs) # 순수 logits 수집
    # #             all_logits.append(logits)
    # #             all_labels.append(labels.to(self.device))
        
    # #     all_logits = torch.cat(all_logits)
    # #     all_labels = torch.cat(all_labels)

    # #     # -- 보정 전 성능 측정 --
    # #     uncalibrated_probs = torch.softmax(all_logits, dim=1)
    # #     uncalibrated_acc = self.accuracy(uncalibrated_probs, all_labels)
    # #     uncalibrated_loss = self.criterion(all_logits, all_labels)
    # #     uncalibrated_ece = self.ece(uncalibrated_probs, all_labels)

    # #     print("\n--- Uncalibrated Model Performance ---")
    # #     print(f"  Accuracy: {uncalibrated_acc.item():.4f}")
    # #     print(f"  NLL Loss: {uncalibrated_loss.item():.4f}")
    # #     print(f"  ECE     : {uncalibrated_ece.item():.4f}")

    # #     # -- 보정 후 성능 측정 --
    # #     calibrated_logits = all_logits / self.temperature
    # #     calibrated_probs = torch.softmax(calibrated_logits, dim=1)
    # #     calibrated_acc = self.accuracy(calibrated_probs, all_labels)
    # #     calibrated_loss = self.criterion(calibrated_logits, all_labels)
    # #     calibrated_ece = self.ece(calibrated_probs, all_labels)

    # #     print("\n--- Calibrated Model Performance ---")
    # #     print(f"  (Applied Temperature: {self.temperature.item():.4f})")
    # #     print(f"  Accuracy: {calibrated_acc.item():.4f}")
    # #     print(f"  NLL Loss: {calibrated_loss.item():.4f}")
    # #     print(f"  ECE     : {calibrated_ece.item():.4f}")

    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.phase1_epochs:
            print(f"\n--- Epoch {self.current_epoch}: Switching to Fine-Tuning Phase (Rank {self.global_rank}) ---")
            
            # 1. 모델 전체 파라미터 동결 해제
            for param in self.model.parameters():
                param.requires_grad = True
            
            # 2. Optimizer가 모델 전체 파라미터를 학습하도록 상태 갱신
            optimizer = self.optimizers()
            # 기존 파라미터 그룹을 비우고,
            optimizer.param_groups.clear()
            # 전체 파라미터를 새로운 학습률과 함께 추가
            optimizer.add_param_group({
                'params': self.parameters(), 
                'lr': self.hparams.learning_rate_phase2
            })
            print(f"Optimizer reconfigured with all parameters and new LR: {self.hparams.learning_rate_phase2} (Rank {self.global_rank})")

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
        
        # ⭐️ [수정 2] 분산 학습 시 로그 동기화를 위해 sync_dist=True 추가
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

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