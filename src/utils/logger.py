import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def get_wandb_logger(config):
    """
    Initializes and returns a Weights & Biases logger based on config.
    """
    if not config['wandb']['enabled']:
        return None
        
    wandb_logger = WandbLogger(
        project=config['project_name'],
        name=config['experiment_name'],
        log_model=config['wandb'].get('log_model', False)
    )
    return wandb_logger
