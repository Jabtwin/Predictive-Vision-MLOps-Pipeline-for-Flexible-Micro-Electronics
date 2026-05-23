import os
import yaml
import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.grasp_unet import GraspUNet
from src.data_pipeline.dataset import FlexibleGraspDataset
from src.data_pipeline.augmentations import get_train_transforms, get_val_transforms
from src.train import GraspLightningModule

def objective(trial):
    # 1. Load base config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Disable W&B for raw HPO unless using W&B Sweeps specifically. 
    # Here we demonstrate native Optuna logging.
    config['wandb']['enabled'] = False

    # 2. Suggest Hyperparameters
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config['training']['weight_decay'] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    config['data']['batch_size'] = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # 3. Setup Data
    img_size = config['data']['img_size']
    # Use smaller mock size for faster trials
    train_dataset = FlexibleGraspDataset(size=50, img_size=img_size, transforms=get_train_transforms(img_size))
    val_dataset = FlexibleGraspDataset(size=10, img_size=img_size, transforms=get_val_transforms(img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=0)

    # 4. Initialize Model & Trainer
    model = GraspLightningModule(config)
    
    trainer = pl.Trainer(
        logger=False,
        max_epochs=3, # Fewer epochs for HPO
        accelerator='auto',
        devices=1,
        enable_checkpointing=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )

    # 5. Run Training
    trainer.fit(model, train_loader, val_loader)

    # 6. Return metric to optimize
    return trainer.callback_metrics["val_loss"].item()

def main():
    print("Starting Optuna Hyperparameter Optimization Study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
