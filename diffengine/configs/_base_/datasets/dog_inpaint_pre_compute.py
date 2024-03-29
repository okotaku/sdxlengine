import torchvision
from mmengine.dataset import InfiniteSampler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.datasets import HFDreamBoothDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    ComputeTimeIds,
    DumpImage,
    GetMaskedImage,
    LoadMask,
    MaskToTensor,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    SaveImageShape,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CheckpointHook,
    CompileHook,
    MemoryFormatHook,
    VisualizationHook,
)

train_pipeline = [
    dict(type=SaveImageShape),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=1024, interpolation="bilinear"),
    dict(type=RandomCrop, size=1024),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(
        type=LoadMask,
        mask_mode="bbox",
        mask_config=dict(
            max_bbox_shape=(256, 256),
            max_bbox_delta=40,
            min_margin=20)),
    dict(type=ComputeTimeIds),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=MaskToTensor),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=GetMaskedImage),
    dict(type=PackInputs,
         input_keys=["img", "mask", "masked_image", "prompt_embeds",
                     "pooled_prompt_embeds", "time_ids"]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=HFDreamBoothDatasetPreComputeEmbs,
        dataset="diffusers/dog-example",
        instance_prompt="a photo of sks dog",
        model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
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
        prompt=["a photo of sks dog"] * 4,
        image=["https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b"] * 4,  # noqa
        mask=["https://github.com/okotaku/diffengine/assets/24734142/d0de4fb9-9183-418a-970d-582e9324f05d"] * 4,  # noqa
        by_epoch=False,
        width=1024,
        height=1024,
        interval=100),
    dict(type=CheckpointHook),
    dict(type=MemoryFormatHook),
    dict(type=CompileHook),
]
