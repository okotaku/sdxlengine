from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_50e_baseline import *
