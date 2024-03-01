from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.models.editors import StableDiffusionXL

base_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
model = dict(type=StableDiffusionXL,
             model=base_model,
             tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer",
                            use_fast=False),
             tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer_2",
                            use_fast=False),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                               subfolder="text_encoder"),
             text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               subfolder="text_encoder_2"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
            unet_lora_config = dict(
                    type=LoraConfig, r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                    type=LoraConfig, r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
