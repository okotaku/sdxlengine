# Playground v2.5

[Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation](https://playground.com/blog/playground-v2-5)

## Abstract

In this work, we share three insights for achieving state-of-the-art aesthetic quality in text-to-image generative models. We focus on three critical aspects for model improvement: enhancing color and contrast, improving generation across multiple aspect ratios, and improving human-centric fine details. First, we delve into the significance of the noise schedule in training a diffusion model, demonstrating its profound impact on realism and visual fidelity. Second, we address the challenge of accommodating various aspect ratios in image generation, emphasizing the importance of preparing a balanced bucketed dataset. Lastly, we investigate the crucial role of aligning model outputs with human preferences, ensuring that generated images resonate with human perceptual expectations. Through extensive analysis and experiments, Playground v2.5 demonstrates state-of-the-art performance in terms of aesthetic quality under various conditions and aspect ratios, outperforming both widely-used open-source models like SDXL \[28\] and Playground v2 \[20\], and closed-source commercial systems such as DALL·E 3 \[2\] and Midjourney v5.2. Our model is open-source, and we hope the development of Playground v2.5 provides valuable guidelines for researchers aiming to elevate the aesthetic quality of diffusion-based image generation models.

## Citation

```
@misc{li2024playground,
      title={Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation},
      author={Daiqing Li and Aleks Kamko and Ehsan Akhgari and Ali Sabet and Linmiao Xu and Suhail Doshi},
      year={2024},
      eprint={2402.17245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train playgroundv2_5_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/playgroundv2_5_pokemon_blip/step41650'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.bfloat16)
pipe = DiffusionPipeline.from_pretrained(
    'playgroundv2_5_pokemon_blip', unet=unet, torch_dtype=torch.bfloat16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]
image.save('demo.png')
```

## Results Example

#### playgroundv2_5_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/40e52983-02ef-44af-a8ec-a39ec3aa95c8)
