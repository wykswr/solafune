from lightning.pytorch import Trainer
from data import TiffDataModule
from model import PepperNet
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
from pathlib import Path
import json

with (Path(__file__).parent/'setting.json').open() as f:
    config = json.load(f)

CKPT = Path(__file__).parent/"ckpt"
date_time = datetime.now().strftime("%m-%d-%Y_%H-%M")
# torch.set_float32_matmul_precision('high')

wandb_logger = WandbLogger(project='salofune_ft')
checkpoint_callback = ModelCheckpoint(
    dirpath=CKPT/date_time,
    monitor='val_loss',
    filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

mod = PepperNet(lr=2e-5, weight_decay=1e-2, warmup_steps=0)

data_module = TiffDataModule(config['train'], 16, 4)

trainer = Trainer(
    max_epochs=12,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=5,
)


def train():
    trainer.fit(mod, datamodule=data_module)


if __name__ == '__main__':
    train()
