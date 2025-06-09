# inference_ensemble.py

import torch
import argparse
import os
import yaml
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from glob import glob

import models # 모델 클래스를 불러오기 위해 필요

# 단일 이미지 또는 폴더 내 이미지 추론을 위한 간단한 데이터셋 클래스
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image), self.image_paths[idx]

def main(args):
    # 0. 설정 로드 및 장치 설정
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. K개의 모델 로드
    # 체크포인트 디렉토리에서 'best.ckpt' 또는 유사한 패턴의 파일들을 모두 찾습니다.
    # ModelCheckpoint에서 filename을 지정했다면 해당 패턴을 사용해야 합니다.
    ckpt_paths = sorted(glob(os.path.join(args.ckpt_dir, 'fold_*', '*.ckpt')))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in {args.ckpt_dir}/fold_*/")
    
    print(f"Found {len(ckpt_paths)} models for ensemble:")
    for path in ckpt_paths:
        print(f"- {path}")

    model_name = config['model'].pop('module_name')
    Model = getattr(models, model_name)
    
    ensemble_models = []
    for ckpt_path in ckpt_paths:
        # load_from_checkpoint는 모델 구조와 가중치를 모두 불러옵니다.
        model = Model.load_from_checkpoint(checkpoint_path=ckpt_path)
        model.to(device)
        model.eval() # 추론 모드로 설정
        ensemble_models.append(model)

    # 2. 추론할 데이터 준비
    # 모델의 이미지 크기 가져오기
    model_variant = config['model']['model_variant']
    _, _, image_size = models.MODEL_MAP[model_variant]

    # 검증/테스트와 동일한 변환 적용 (Augmentation 없음)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_paths = []
    if os.path.isdir(args.image_path):
        image_paths = sorted([os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))])
    else:
        image_paths = [args.image_path]

    dataset = InferenceDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 클래스 이름 정보 로드 (결과 해석을 위해)
    # DataModule을 임시로 생성하여 class_to_idx를 얻는 방법
    data_dir = config['data']['data_dir']
    temp_dataset = ImageFolder(data_dir)
    idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}

    # 3. 앙상블 추론 실행
    results = {}
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            
            # 각 모델의 예측 확률(softmax)을 저장할 리스트
            batch_probas = []
            
            for model in ensemble_models:
                logits = model(images)
                probas = F.softmax(logits, dim=1)
                batch_probas.append(probas.cpu())
            
            # K개 모델의 예측 확률을 평균 (Soft Voting)
            # (K, Batch, Num_Classes) 텐서로 변환 후 평균
            avg_probas = torch.stack(batch_probas).mean(dim=0)
            
            # 최종 예측 클래스
            final_preds = torch.argmax(avg_probas, dim=1)
            
            # 결과 저장
            for path, pred_idx in zip(paths, final_preds):
                class_name = idx_to_class[pred_idx.item()]
                results[os.path.basename(path)] = class_name
    
    # 4. 결과 출력
    print("\n===== Ensemble Inference Results =====")
    for filename, class_name in results.items():
        print(f"File: {filename}, Predicted Class: {class_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ensemble inference using K-Fold models.")
    parser.add_argument('--config', type=str, required=True, help="Path to the same config file used for training.")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Directory containing the fold-specific checkpoint folders (e.g., 'checkpoints/').")
    parser.add_argument('--image_path', type=str, required=True, help="Path to an image or a directory of images to infer.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference.")
    
    args = parser.parse_args()
    main(args)