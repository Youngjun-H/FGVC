import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn.functional as F

from PIL import Image
import os
import pandas as pd # Pandas 추가
from tqdm import tqdm

# --- 1. 기본 설정 및 장치 확인 ---
MODEL_VARIANT = 'B0'
MODEL_PATH = f'efficientnet_{MODEL_VARIANT}_best_final_model_v2.pth'
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
SAMPLE_SUBMISSION_PATH = 'dataset/sample_submission.csv' # 샘플 제출 파일 경로
OUTPUT_CSV_PATH = 'baseline_submission.csv'

NUM_CLASSES = 396
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 # BATCH_SIZE는 CFG 딕셔너리 대신 직접 정의

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. 모델 및 데이터셋 클래스 정의 ---

# 테스트 데이터셋을 위한 커스텀 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image # 테스트셋이므로 이미지만 반환

# --- 3. 클래스 이름 및 전처리 정의 ---

# 클래스 이름 로드
try:
    train_dataset_for_classes = ImageFolder(TRAIN_DIR)
    class_names = train_dataset_for_classes.classes
    print(f"Successfully loaded {len(class_names)} class names.")
except FileNotFoundError:
    print(f"Error: Training directory '{TRAIN_DIR}' not found.")
    exit()

# 검증/테스트용 전처리 정의
val_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. 메인 추론 및 제출 파일 생성 로직 ---
if __name__ == '__main__':
    # 테스트 데이터셋 및 로더 생성
    test_dataset = CustomImageDataset(TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 모델 생성 및 학습된 가중치 로드
    print(f"Loading model from {MODEL_PATH}")

    # 1. BaseModel 래퍼 없이 EfficientNet 모델 구조를 직접 생성합니다.
    model = models.efficientnet_b0(weights=None)
    # 2. 학습 때와 동일하게 분류층을 교체합니다.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )

    # 3. 이제 모델 구조가 저장된 파일과 일치하므로, state_dict를 직접 로드할 수 있습니다.
    try:
        # 저장된 state_dict 로드
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # 'model.' 접두사 제거
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        # 수정된 state_dict 로드
        model.load_state_dict(new_state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        exit()

    model.to(DEVICE)
    
    # 추론 시작
    print("Starting inference...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            # Softmax를 적용하여 확률 계산
            probs = F.softmax(outputs, dim=1)

            # 각 배치의 확률을 딕셔너리 리스트로 변환
            for prob in probs.cpu():  # prob: (num_classes,)
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)
    
    print("Inference finished. Creating submission file...")
    
    # 결과를 DataFrame으로 변환
    pred_df = pd.DataFrame(results)

    # 샘플 제출 파일 로드 및 형식 맞추기
    try:
        submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Sample submission file not found at '{SAMPLE_SUBMISSION_PATH}'")
        # 샘플 파일이 없을 경우, 이미지 파일명과 예측 결과로 DataFrame을 직접 생성
        # 파일 순서가 test_loader 순서와 동일하다고 가정
        submission_df = pd.DataFrame({'ID': [os.path.basename(f) for f in test_dataset.image_files]})
        for col in class_names:
            submission_df[col] = 0 # 빈 컬럼 추가
            
    # 'ID' 컬럼을 제외한 클래스 컬럼 순서를 샘플 파일에 맞춤
    class_columns = submission_df.columns[1:]
    
    # 예측 DataFrame의 컬럼 순서를 샘플과 동일하게 정렬
    pred_df = pred_df[class_columns]

    # 샘플 파일에 예측값 채우기
    submission_df[class_columns] = pred_df.values
    
    # 최종 제출 파일 저장
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Successfully created submission file at '{OUTPUT_CSV_PATH}'")