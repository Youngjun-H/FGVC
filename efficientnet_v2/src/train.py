import yaml
import argparse
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import EfficientNetFineTuner
from dotenv import load_dotenv
import torch
import os

import data_modules
import models
from models import MODEL_MAP # 모델별 이미지 크기 정보를 가져오기 위해 임포트

def main(config_path: str):
    # Enable Tensor Core usage for better performance
    torch.set_float32_matmul_precision('high')
    
    # 1. YAML 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ⭐️ 2. W&B Logger 생성
    # config 파일에 'wandb' 섹션이 있을 경우에만 WandbLogger를 생성합니다.
    logger = None
    if 'wandb' in config:
        # now 변수를 동적으로 처리하기 위해 pytorch_lightning의 기능을 잠시 사용
        from lightning.pytorch.utilities.rank_zero import rank_zero_only
        
        @rank_zero_only
        def resolve_wandb_name():
            from datetime import datetime
            if "name" in config["wandb"] and "${now" in config["wandb"]["name"]:
                config["wandb"]["name"] = config["wandb"]["name"].replace("${now:%Y-%m-%d_%H-%M-%S}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        resolve_wandb_name()
        
        logger = WandbLogger(**config['wandb'])

    # 2. DataModule 인스턴스화
    data_config = config['data']
    data_module_name = data_config.pop('module_name')
    # 모델에 맞는 이미지 크기를 config에서 가져와 DataModule에 전달
    model_variant = config['model']['model_variant']
    _, _, image_size = MODEL_MAP[model_variant]
    DataModule = getattr(data_modules, data_module_name)
    datamodule = DataModule(**data_config, image_size=image_size)

    # 3. Model 인스턴스화
    model_config = config['model']
    model_name = model_config.pop('module_name')
    Model = getattr(models, model_name)
    model = Model(**model_config)

    # 4. 콜백(Callbacks) 설정
    callbacks = []
    if 'early_stopping' in config['callbacks']:
        early_stop_callback = EarlyStopping(**config['callbacks']['early_stopping'])
        callbacks.append(early_stop_callback)
    if 'model_checkpoint' in config['callbacks']:
        checkpoint_callback = ModelCheckpoint(**config['callbacks']['model_checkpoint'])
        callbacks.append(checkpoint_callback)

    # 5. Trainer 인스턴스화 및 학습 시작    
    trainer = L.Trainer(**config['trainer'], callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule)

    # =================================================================
    # ⭐️ 2. 학습 완료 후, 모델 보정 단계 추가
    # =================================================================
    print("\n--- Starting Model Calibration (Temperature Scaling) ---")

    # 가장 성능이 좋았던 체크포인트 경로 가져오기
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path:
        print("No best checkpoint found. Skipping calibration.")
    else:
        print(f"Loading best model from: {best_ckpt_path}")
        # 체크포인트로부터 모델 로드
        calibrated_model = EfficientNetFineTuner.load_from_checkpoint(best_ckpt_path)
        
        # 검증 데이터로더 준비
        # datamodule.setup('fit')을 호출하여 val_dataset이 준비되도록 보장
        datamodule.setup('fit')
        val_loader = datamodule.val_dataloader()
        
        # 최적 온도 찾기 메서드 호출
        calibrated_model.find_optimal_temperature(val_loader=val_loader)

        # 보정된 모델 저장
        calibrated_model_path = best_ckpt_path.replace(".ckpt", "_calibrated.ckpt")
        print(f"Saving calibrated model to: {calibrated_model_path}")
        trainer.save_checkpoint(calibrated_model_path, weights_only=True)
        # 만약 전체 모델을 저장하고 싶다면:
        # torch.save(calibrated_model.state_dict(), calibrated_model_path)

    if logger:
        logger.experiment.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an EfficientNet model using PyTorch Lightning.")
    parser.add_argument('--config', type=str, default='configs/efficientnet_b0_config.yaml', help="Path to the config file.")
    args = parser.parse_args()
    load_dotenv()

    main(args.config)