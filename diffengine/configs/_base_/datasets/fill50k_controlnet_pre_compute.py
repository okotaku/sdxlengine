import torchvision
from mmengine.dataset import DefaultSampler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.datasets import HFConditionDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    DumpImage,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CompileHook,
    ControlNetSaveHook,
    VisualizationHook,
)

train_pipeline = [
    dict(type=SaveImageShape),
    dict(
        type=TorchVisonTransformWrapper,
        transform=torchvision.transforms.Resize,
        size=1024,
        interpolation="bilinear",
        keys=["img", "condition_img"]),
    dict(type=RandomCrop, size=1024, keys=["img", "condition_img"]),
    dict(type=RandomHorizontalFlip, p=0.5, keys=["img", "condition_img"]),
    dict(type=ComputeTimeIds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor,
         keys=["img", "condition_img"]),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs,
         input_keys=[
            "img",
            "condition_img",
            "time_ids",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "time_ids",
        ]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=HFConditionDatasetPreComputeEmbs,
        dataset="fusing/fill50k",
        condition_column="conditioning_image",
        caption_column="text",
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
        proportion_empty_prompts=0.1,
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["cyan circle with brown floral background"] * 4,
        condition_image=[
            'https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191'  # noqa
        ] * 4),
    dict(type=ControlNetSaveHook),
    dict(type=CompileHook),
]
