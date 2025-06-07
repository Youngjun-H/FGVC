import torch
from torch import nn
import torch.optim as optim
import lightning as L
from torchmetrics import Accuracy
from config import MODEL_LOADER, WEIGHTS

class EfficientNetLightning(L.LightningModule):
    def __init__(self, num_classes, learning_rate, phase=1):
        super().__init__()
        self.save_hyperparameters()
        
        # Load base model
        self.model = MODEL_LOADER(weights=WEIGHTS)
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.learning_rate = learning_rate
        self.phase = phase  # 1 for top layers, 2 for fine-tuning
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        # Log training metrics
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        # Log validation metrics
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if self.phase == 1:
            # Phase 1: Only train classifier
            optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        else:
            # Phase 2: Train all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        } 