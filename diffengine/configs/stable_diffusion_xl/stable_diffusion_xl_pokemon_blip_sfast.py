from mmengine.config import read_base

from diffengine.engine.hooks import SFastHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(weight_dtype="bf16")

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=CheckpointHook),
    dict(type=MemoryFormatHook),
    dict(type=SFastHook),
]
