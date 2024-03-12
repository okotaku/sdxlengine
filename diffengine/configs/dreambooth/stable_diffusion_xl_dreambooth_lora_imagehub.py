from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagehub_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_lora import *
    from .._base_.schedules.stable_diffusion_1k import *

model.update(weight_dtype="bf16")

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=FusedAdam, lr=1e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))
