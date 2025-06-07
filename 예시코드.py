import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # 학습 진행 상황 시각화
from collections import defaultdict

# --- 1. 기본 설정 및 장치 확인 ---
TRAIN_DIR = 'train'  # 실제 학습 데이터 폴더 경로로 수정
# TEST_DIR = 'test'    # 실제 평가 데이터 폴더 경로로 수정
NUM_CLASSES = 396
VALIDATION_SPLIT = 0.2

# 모델별 설정 (예시: EfficientNetB0)
MODEL_VARIANT = 'B0' # 'B0', 'B1', 'B2', 'B3' 중 선택
if MODEL_VARIANT == 'B0':
    IMG_SIZE = (224, 224)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    EfficientNet_model_loader = models.efficientnet_b0
elif MODEL_VARIANT == 'B1':
    IMG_SIZE = (240, 240)
    weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    EfficientNet_model_loader = models.efficientnet_b1
elif MODEL_VARIANT == 'B2':
    IMG_SIZE = (260, 260)
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    EfficientNet_model_loader = models.efficientnet_b2
elif MODEL_VARIANT == 'B3':
    IMG_SIZE = (300, 300)
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    EfficientNet_model_loader = models.efficientnet_b3
else:
    raise ValueError("Unsupported EfficientNet variant")

BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 50
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5
EARLY_STOPPING_PATIENCE = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. 데이터 로드 및 전처리 ---
# torchvision 모델은 [-1, 1] 정규화가 아닌 ImageNet 통계량 정규화를 사용
# weights.transforms()가 자동으로 정규화 및 크기 조정을 처리
train_transforms = T.Compose([
    T.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
    T.RandomRotation(degrees=20),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 전체 데이터셋 로드
full_dataset = ImageFolder(TRAIN_DIR)

# 데이터셋 분할
num_train = len(full_dataset)
val_size = int(VALIDATION_SPLIT * num_train)
train_size = num_train - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 각 데이터셋에 맞는 transform 적용
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

train_dataset = TransformedDataset(train_dataset, transform=train_transforms)
val_dataset = TransformedDataset(val_dataset, transform=val_transforms)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class_names = full_dataset.classes
# print(f"Found {len(class_names)} classes.")

# --- 3. 모델 구축 ---
def build_model(num_classes):
    model = EfficientNet_model_loader(weights=weights)
    
    # 기본 모델의 가중치 동결
    for param in model.parameters():
        param.requires_grad = False
        
    # 분류층 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

model = build_model(NUM_CLASSES)
# print(model)

# --- 4. 학습/검증 함수 및 메인 루프 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

# Main training loop
history = defaultdict(list)
best_val_acc = 0.0
epochs_no_improve = 0
checkpoint_filepath = f'efficientnet_{MODEL_VARIANT}_best_model.pth'

# --- 5. 학습 - 1단계: 상위 분류층 학습 ---
print(f"\n--- STARTING PHASE 1 TRAINING (Training top layers) for EfficientNet{MODEL_VARIANT} ---")
# 새로 추가한 분류층의 파라미터만 옵티마이저에 전달
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE_PHASE1)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS_PHASE1):
    print(f"Epoch {epoch+1}/{EPOCHS_PHASE1}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    history['train_loss_p1'].append(train_loss)
    history['train_acc_p1'].append(train_acc)
    history['val_loss_p1'].append(val_loss)
    history['val_acc_p1'].append(val_acc)
    
    if val_acc > best_val_acc:
        print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_filepath)

# --- 6. 학습 - 2단계: 미세 조정 ---
print(f"\n--- STARTING PHASE 2 TRAINING (Fine-tuning) for EfficientNet{MODEL_VARIANT} ---")
# 전체 모델 파라미터 동결 해제
for param in model.parameters():
    param.requires_grad = True

# 모델을 다시 로드하여 최상의 상태에서 시작
model.load_state_dict(torch.load(checkpoint_filepath))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2) # 매우 낮은 학습률
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

for epoch in range(EPOCHS_PHASE2):
    print(f"Epoch {epoch+1+EPOCHS_PHASE1}/{EPOCHS_PHASE1+EPOCHS_PHASE2}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    history['train_loss_p2'].append(train_loss)
    history['train_acc_p2'].append(train_acc)
    history['val_loss_p2'].append(val_loss)
    history['val_acc_p2'].append(val_acc)
    
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_filepath)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation accuracy did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break
        
# --- 7. 학습 곡선 시각화 ---
def plot_history(history, model_name=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    train_acc_full = history['train_acc_p1'] + history['train_acc_p2']
    val_acc_full = history['val_acc_p1'] + history['val_acc_p2']
    train_loss_full = history['train_loss_p1'] + history['train_loss_p2']
    val_loss_full = history['val_loss_p1'] + history['val_loss_p2']
    
    epochs = range(1, len(train_acc_full) + 1)

    ax1.plot(epochs, train_acc_full, 'bo-', label='Training Acc')
    ax1.plot(epochs, val_acc_full, 'ro-', label='Validation Acc')
    ax1.set_title(f'{model_name} Model Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(epochs, train_loss_full, 'bo-', label='Training Loss')
    ax2.plot(epochs, val_loss_full, 'ro-', label='Validation Loss')
    ax2.set_title(f'{model_name} Model Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_pytorch_{model_name}.png')
    plt.show()

plot_history(history, f"EfficientNet{MODEL_VARIANT}")

# --- 8. 모델 평가 (Test Set) ---
# 최적 가중치를 가진 모델 생성 및 로드
final_model = build_model(NUM_CLASSES)
final_model.load_state_dict(torch.load(checkpoint_filepath))
final_model.to(DEVICE)

# 테스트 데이터셋 로더 (폴더 구조가 train과 동일하다고 가정)
# test_dataset = ImageFolder(TEST_DIR, transform=val_transforms)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# 
# print("\n--- EVALUATING ON TEST SET ---")
# test_loss, test_acc = validate(final_model, test_loader, criterion, DEVICE)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
#
# # 추가적으로 scikit-learn을 이용한 상세 리포트
# from sklearn.metrics import classification_report, confusion_matrix
# y_pred = []
# y_true = []
# final_model.eval()
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#         outputs = final_model(inputs)
#         _, preds = torch.max(outputs, 1)
#         y_pred.extend(preds.cpu().numpy())
#         y_true.extend(labels.cpu().numpy())
#
# print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print(f"Training and evaluation for EfficientNet{MODEL_DEVIATION} complete. Best model saved to {checkpoint_filepath}")