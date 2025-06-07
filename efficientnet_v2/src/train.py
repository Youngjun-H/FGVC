import yaml
import argparse
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import data_modules
import models
from models import MODEL_MAP # 모델별 이미지 크기 정보를 가져오기 위해 임포트

def main(config_path: str):
    # 1. YAML 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    trainer = L.Trainer(**config['trainer'], callbacks=callbacks)
    trainer.fit(model, datamodule)

    # (선택) 학습 후 테스트 실행
    # print("\n--- Starting Test Phase ---")
    # trainer.test(model, datamodule, ckpt_path='best') # 가장 좋았던 체크포인트로 테스트

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an EfficientNet model using PyTorch Lightning.")
    parser.add_argument('--config', type=str, default='configs/efficientnet_b0_config.yaml', help="Path to the config file.")
    args = parser.parse_args()
    
    main(args.config)