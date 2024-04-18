import lightning as L
from Prithvi_100M.Prithvi_tool import load_encoder
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import get_cosine_schedule_with_warmup


class PepperNet(L.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=0.01, warmup_steps=32):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = load_encoder().train()
        self.fc = nn.Linear(768, 2)
        # xavier init
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # metrics
        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.prec = BinaryPrecision()
        self.rec = BinaryRecall()
        self.auc = BinaryAUROC()


    def forward(self, x):
        """
        input: x (batch, 6, 1, 224, 224)
        output: y (batch, 1)
        """
        latent = self.backbone(x)
        x = latent[:, 0, :] # CLS
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        y_hat = self(batch['image'])
        y = batch['label']
        # weight = torch.tensor([0.206119, 0.793881]).to(y.device)
        # loss = F.cross_entropy(y_hat, y, weight=weight, label_smoothing=0.1)
        loss = F.cross_entropy(y_hat, y)
        
        self.log('train_loss', loss, sync_dist=True,
                 batch_size=batch["image"].size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch["image"].size(0)
        y_hat = self(batch['image'])
        y = batch['label']
        # weight = torch.tensor([0.206119, 0.793881]).to(y.device)
        # loss = F.cross_entropy(y_hat, y, weight=weight, label_smoothing=0.1)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True,
                 batch_size=batch_size)
        y_hat = F.softmax(y_hat, dim=1)[:,-1]
        self.acc(y_hat, y)
        self.prec(y_hat, y)
        self.rec(y_hat, y)
        self.f1(y_hat, y)
        self.auc(y_hat, y)
        self.log('val_acc', self.acc, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log('val_precision', self.prec, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log('val_recall', self.rec, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log('val_f1', self.f1, sync_dist=True, batch_size=batch_size, on_epoch=True)
        self.log('val_auc', self.auc, sync_dist=True, batch_size=batch_size, on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, total_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }