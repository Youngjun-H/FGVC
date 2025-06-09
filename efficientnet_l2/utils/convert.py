import torch
from efficientnet_v2.src.models import EfficientNetFineTuner

best_ckpt_path = "efficientnet_v2/outputs/checkpoints/efficientnet_b0/efficientnet-b0-60-0.9291.ckpt"

loaded_model = EfficientNetFineTuner.load_from_checkpoint(best_ckpt_path)
print("체크포인트로부터 모델을 성공적으로 불러왔습니다.")

# 4. 모델의 state_dict를 .pth 파일로 저장합니다.
pth_save_path = "temp.pth"
torch.save(loaded_model.state_dict(), pth_save_path)

print(f"모델의 가중치가 '{pth_save_path}' 경로에 .pth 파일로 저장되었습니다.")