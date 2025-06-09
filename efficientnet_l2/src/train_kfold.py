# train_kfold.py

import yaml
import argparse
import os
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder
from dotenv import load_dotenv
from datetime import datetime

import data_modules
import models
from models import MODEL_MAP
import wandb

def main(config_path: str):
    # Enable Tensor Core usage
    torch.set_float32_matmul_precision('high')
    
    # 1. YAML 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # K-Fold 설정
    k_folds = config.get('k_folds', 5)  # 설정 파일에 없으면 기본값 5 사용
    
    # 2. 전체 데이터셋 정보 로드 (인덱스 분할을 위해)
    # DataModule 설정에서 데이터 경로 가져오기
    data_dir = config['data']['data_dir']
    # ImageFolder를 사용하여 파일 경로와 레이블 목록 생성
    full_dataset = ImageFolder(data_dir)
    image_paths = [path for path, _ in full_dataset.imgs]
    labels = full_dataset.targets
    
    # Stratified K-Fold 분할기 생성 (레이블 분포를 유지하며 분할)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.get('seed', 42))

    # 각 Fold의 검증 점수를 저장할 리스트
    val_scores = []

    # 3. K-Fold 루프 시작
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_dataset)), labels)):
        print(f"\n===== Fold {fold+1}/{k_folds} =====")

        # --- 각 Fold마다 새로운 인스턴스 생성 ---

        # 3-1. W&B Logger 생성 (Fold별로 구분)
        logger = None
        if 'wandb' in config:
            wandb_config = config['wandb'].copy()
        
        # 1. YAML에서 name 템플릿을 가져옵니다.
        name_template = wandb_config.get('name', "kfold-run-${now:%Y-%m-%d_%H-%M-%S}")

        # 2. ${now...} 플레이스홀더가 있으면 현재 시간으로 치환합니다.
        if "${now" in name_template:
            current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # 플레이스홀더 부분을 실제 시간으로 교체하여 그룹 이름을 생성합니다.
            group_name = name_template.replace("${now:%Y-%m-%d_%H-%M-%S}", current_time_str)
        else:
            # 플레이스홀더가 없는 경우, 설정된 이름을 그대로 그룹 이름으로 사용합니다.
            group_name = name_template

        # 3. W&B에서 실행들을 묶어줄 'group'을 설정합니다.
        wandb_config['group'] = group_name
        
        # 4. 현재 Fold의 실제 실행 'name'을 설정합니다. (예: my-run-2025-06-08_23-55-00_fold_1)
        wandb_config['name'] = f"{group_name}_fold_{fold+1}"

        # 5. log_model 설정이 YAML에 없으면 기본값 False를 사용하도록 합니다.
        wandb_config.setdefault('log_model', False)
        
        logger = WandbLogger(**wandb_config)
            
        # 3-2. DataModule 인스턴스화 (Fold에 맞는 인덱스 전달)
        data_config = config['data'].copy()
        data_module_name = data_config.pop('module_name')
        model_variant = config['model']['model_variant']
        _, _, image_size = MODEL_MAP[model_variant]
        DataModule = getattr(data_modules, data_module_name)
        # 수정된 DataModule에 train/val 인덱스 전달
        datamodule = DataModule(**data_config, image_size=image_size, train_indices=train_idx, val_indices=val_idx)

        # 3-3. Model 인스턴스화 (매번 새로운 모델로 시작)
        model_config = config['model'].copy()
        model_name = model_config.pop('module_name')
        Model = getattr(models, model_name)
        model = Model(**model_config)

        # 3-4. 콜백 설정 (체크포인트 경로를 Fold별로 지정)
        callbacks = []
        checkpoint_config = config['callbacks']['model_checkpoint'].copy()
        # 저장 경로에 fold 번호 추가
        original_dir = checkpoint_config.get('dirpath', 'checkpoints/')
        checkpoint_config['dirpath'] = os.path.join(original_dir, f"fold_{fold}")
        
        if 'early_stopping' in config['callbacks']:
            callbacks.append(EarlyStopping(**config['callbacks']['early_stopping']))
        callbacks.append(ModelCheckpoint(**checkpoint_config))

        # 3-5. Trainer 인스턴스화 및 학습 시작
        trainer = L.Trainer(**config['trainer'], callbacks=callbacks, logger=logger)
        trainer.fit(model, datamodule=datamodule)
        
        # 3-6. 현재 Fold의 최적 모델로 검증 성능 측정
        # best_model_path를 사용하여 최적 모델의 성능을 기록합니다.
        # ModelCheckpoint의 `monitor` 지표를 사용합니다.
        val_result = trainer.validate(ckpt_path='best', datamodule=datamodule)
        score_key = f"{config['callbacks']['model_checkpoint']['monitor'].split('/')[-1]}" # 예: 'val_acc'
        val_scores.append(val_result[0][score_key])

        if logger:
            # W&B 실행 종료
            wandb.finish()
    
    # 4. 최종 결과 출력
    print(f"\n===== K-Fold Cross-Validation Results =====")
    for i, score in enumerate(val_scores):
        print(f"Fold {i+1} Validation Score: {score:.4f}")
    print(f"\nAverage Validation Score: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model using K-Fold cross-validation.")
    parser.add_argument('--config', type=str, default='configs/efficientnet_v2l_config.yaml', help="Path to the config file.")
    args = parser.parse_args()
    load_dotenv()
    
    # W&B 로그인 (필요 시)
    # import wandb
    # wandb.login()

    main(args.config)