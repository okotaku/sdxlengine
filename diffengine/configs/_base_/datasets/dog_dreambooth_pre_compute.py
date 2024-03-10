import torchvision
from mmengine.dataset import InfiniteSampler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.datasets import HFDreamBoothDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CompileHook,
    MemoryFormatHook,
    PeftSaveHook,
    VisualizationHook,
)

train_pipeline = [
    dict(type=SaveImageShape),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=1024, interpolation="bilinear"),
    dict(type=RandomCrop, size=1024),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=ComputeTimeIds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs,
         input_keys=[
            "img",
            "time_ids",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDreamBoothDatasetPreComputeEmbs,
        dataset="diffusers/dog-example",
        instance_prompt="a photo of sks dog",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                    subfolder="tokenizer",
                    use_fast=False),
        tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                    subfolder="tokenizer_2",
                    use_fast=False),
        text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                        subfolder="text_encoder"),
        text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                        subfolder="text_encoder_2"),
        pipeline=train_pipeline),
    sampler=dict(type=InfiniteSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["A photo of sks dog in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
    dict(type=MemoryFormatHook),
    dict(type=CompileHook),
]
