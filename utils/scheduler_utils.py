from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from omegaconf import DictConfig

def get_scheduler(optimizer, scheduler_config: DictConfig):
    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config.get('params', {})

    if scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")