import yaml
import argparse
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import EfficientNetFineTuner
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from torchvision.datasets import ImageFolder
import torch
import os

import data_modules
import models
from models import MODEL_MAP # 모델별 이미지 크기 정보를 가져오기 위해 임포트

def main(config_path: str):
    # Enable Tensor Core usage for better performance
    torch.set_float32_matmul_precision('high')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = substitute_env_vars(config)

    # 1. 전체 데이터셋 로드
    full_dataset = ImageFolder(config['data']['data_dir'])
    
    # 2. K-Fold 설정
    kf = KFold(n_splits=config['k_fold']['n_splits'], shuffle=True, random_state=42)

    # 3. K-Fold 루프 시작
    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        print(f"\n===== STARTING FOLD {fold+1}/{config['k_fold']['n_splits']} =====")
        
        # -- 각 폴드마다 새로운 인스턴스 생성 --
        # Logger (WandB 그룹 사용)
        wandb_logger = WandbLogger(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            group=config['wandb']['group'],
            name=f"fold-{fold+1}"
        )

        # Checkpoint 콜백 (폴드별로 다른 경로에 저장)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./outputs/fold_{fold+1}/",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename="best_model"
        )
        
        # DataModule (현재 폴드의 인덱스 전달)
        _, _, image_size = MODEL_MAP[config['model']['model_variant']]
        datamodule = data_modules.ImageFolderDataModule(
            **config['data'], 
            image_size=image_size,
            train_indices=train_indices,
            val_indices=val_indices
        )

        # Model
        model = models.EfficientNetFineTuner(**config['model'])

        # Trainer
        trainer = L.Trainer(
            **config['trainer'],
            callbacks=[checkpoint_callback, EarlyStopping(...)],
            logger=wandb_logger
        )

        # 학습 시작
        trainer.fit(model, datamodule)
        wandb_logger.experiment.finish()
        
    print("\n===== K-FOLD TRAINING COMPLETE =====")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an EfficientNet model using PyTorch Lightning.")
    parser.add_argument('--config', type=str, default='configs/efficientnet_b0_config.yaml', help="Path to the config file.")
    args = parser.parse_args()
    load_dotenv()

    main(args.config)