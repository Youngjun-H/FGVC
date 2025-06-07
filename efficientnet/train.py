import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import (
    TRAIN_DIR, NUM_CLASSES, BATCH_SIZE,
    EPOCHS_PHASE1, EPOCHS_PHASE2,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
    EARLY_STOPPING_PATIENCE, MODEL_VARIANT
)
from data import EfficientNetDataModule
from model import EfficientNetLightning

def main():
    # Initialize data module
    data_module = EfficientNetDataModule(
        data_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Setup logging
    logger = TensorBoardLogger("lightning_logs", name=f"efficientnet_{MODEL_VARIANT}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/efficientnet_{MODEL_VARIANT}",
        filename="{epoch:02d}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val_acc",
        mode="max"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=EARLY_STOPPING_PATIENCE,
        mode="max"
    )
    
    # Phase 1: Train only classifier
    print("\n--- STARTING PHASE 1 TRAINING (Training top layers) ---")
    model_phase1 = EfficientNetLightning(
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE_PHASE1,
        phase=1
    )
    
    trainer_phase1 = L.Trainer(
        max_epochs=EPOCHS_PHASE1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=8
    )
    
    trainer_phase1.fit(model_phase1, data_module)
    
    # Phase 2: Fine-tune entire model
    print("\n--- STARTING PHASE 2 TRAINING (Fine-tuning) ---")
    model_phase2 = EfficientNetLightning(
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE_PHASE2,
        phase=2
    )
    
    # Load best weights from phase 1
    model_phase2.load_state_dict(model_phase1.state_dict())
    
    trainer_phase2 = L.Trainer(
        max_epochs=EPOCHS_PHASE2,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=8
    )
    
    trainer_phase2.fit(model_phase2, data_module)
    
    print(f"\nTraining complete! Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main() 