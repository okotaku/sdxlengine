from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.playgroundv2_5 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(weight_dtype="bf16")
train_cfg = dict(by_epoch=True, max_epochs=1)
