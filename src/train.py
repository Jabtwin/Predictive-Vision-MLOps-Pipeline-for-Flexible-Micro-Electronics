import os
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

# Adjust imports assuming this is run from project root or src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.grasp_unet import GraspUNet
from src.data_pipeline.dataset import FlexibleGraspDataset
from src.data_pipeline.augmentations import get_train_transforms, get_val_transforms
from src.utils.metrics import StressAwareLoss
from src.utils.logger import get_wandb_logger

class GraspLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = GraspUNet(
            in_channels=config['model']['in_channels'],
            out_classes=config['model']['out_classes']
        )
        self.criterion = StressAwareLoss(
            penalty_weight=config['training']['loss_stress_penalty_weight']
        )
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, heatmaps, stress_masks = batch['image'], batch['heatmap'], batch['stress_mask']
        preds = self(images)
        loss, base_loss, stress_penalty = self.criterion(preds, heatmaps, stress_masks)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', base_loss, on_step=True, on_epoch=True)
        self.log('train_stress_penalty', stress_penalty, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, heatmaps, stress_masks = batch['image'], batch['heatmap'], batch['stress_mask']
        preds = self(images)
        loss, base_loss, stress_penalty = self.criterion(preds, heatmaps, stress_masks)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log a sample heatmap to W&B every epoch for visual verification
        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            # Take the first image in the batch
            sample_img = images[0].cpu().numpy().transpose(1, 2, 0)
            sample_pred = preds[0][0].cpu().numpy()
            sample_gt = heatmaps[0][0].cpu().numpy()
            
            # Normalize image back for display (approximate)
            sample_img = (sample_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            self.logger.experiment.log({
                "val_images": [
                    wandb.Image(sample_img, caption="Input"),
                    wandb.Image(sample_gt, caption="Ground Truth Heatmap"),
                    wandb.Image(sample_pred, caption="Predicted Heatmap")
                ]
            })

        return loss

    def configure_optimizers(self):
        optimizer_name = self.hparams['training'].get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        return optimizer

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Datasets and Dataloaders
    img_size = config['data']['img_size']
    train_dataset = FlexibleGraspDataset(
        size=config['data']['mock_dataset_size'], 
        img_size=img_size, 
        transforms=get_train_transforms(img_size)
    )
    val_dataset = FlexibleGraspDataset(
        size=max(10, int(config['data']['mock_dataset_size'] * 0.2)), 
        img_size=img_size, 
        transforms=get_val_transforms(img_size)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=0)

    # Initialize Model
    model = GraspLightningModule(config)

    # Initialize Logger
    # For mock execution, disable W&B to avoid login prompts if not configured, or set mode to offline
    if os.environ.get("WANDB_MODE") == "offline":
        config['wandb']['enabled'] = True
    elif "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not found, disabling W&B logger for mock run.")
        config['wandb']['enabled'] = False

    wandb_logger = get_wandb_logger(config)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        logger=wandb_logger if wandb_logger else False,
        log_every_n_steps=1,
        # Use CPU if no GPU
        accelerator='auto',
        devices=1,
        enable_checkpointing=False, # Disable for mock run
    )

    # Start Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
