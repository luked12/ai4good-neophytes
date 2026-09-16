from pytorch_lightning import callbacks
from omegaconf import DictConfig


def get_callbacks(callbacks_config: DictConfig):
    """Instantiate the callbacks listed in the model config.

    Every entry is ``{type: <class name in pytorch_lightning.callbacks>, params: {...}}``,
    so callbacks can be added or configured without touching train.py.
    """
    callbacks_list = []
    for callback in callbacks_config:
        callback_class = getattr(callbacks, callback['type'])
        callbacks_list.append(callback_class(**callback.get('params', {})))
    return callbacks_list
