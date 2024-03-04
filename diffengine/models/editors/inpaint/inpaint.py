from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.inpaint.data_preprocessor import (
    InpaintDataPreprocessor,
)
from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL


class StableDiffusionXLInpaint(StableDiffusionXL):
    """Stable Diffusion Inpaint.

    Args:
    ----
        model (str): pretrained model name of stable diffusion.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`InpaintDataPreprocessor`.

    """

    def __init__(self,
                 *args,
                 model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": InpaintDataPreprocessor}

        super().__init__(
            *args,
            model=model,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        # Fix input channels of Unet
        in_channels = 9
        if self.unet.in_channels != in_channels:
            out_channels = self.unet.conv_in.out_channels
            self.unet.register_to_config(in_channels=in_channels)

            with torch.no_grad():
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels, self.unet.conv_in.kernel_size,
                    self.unet.conv_in.stride, self.unet.conv_in.padding,
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
                self.unet.conv_in = new_conv_in

        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder_one.gradient_checkpointing_enable()
                self.text_encoder_two.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if (not self.finetune_text_encoder) and (
                not self.pre_compute_text_embeddings):
            self.text_encoder_one.requires_grad_(requires_grad=False)
            self.text_encoder_two.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              image: list[str | Image.Image],
              mask: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int = 1024,
              width: int = 1024,
              num_inference_steps: int = 50,
              output_type: str = "pil",
              seed: int = 0,
              **kwargs) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            image (`List[Union[str, Image.Image]]`):
                The image for inpainting.
            mask (`List[Union[str, Image.Image]]`):
                The mask for inpainting.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int):
                The height in pixels of the generated image. Defaults to 1024.
            width (int):
                The width in pixels of the generated image. Defaults to 1024.
            num_inference_steps (int): Number of inference steps.
                Defaults to 50.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            seed (int): The seed for random number generator.
                Defaults to 0.
            **kwargs: Other arguments.

        """
        assert len(prompt) == len(image) == len(mask)
        if self.pre_compute_text_embeddings:
            pipeline = AutoPipelineForInpainting.from_pretrained(
                self.model,
                vae=self.vae,
                unet=self.unet,
                safety_checker=None,
                torch_dtype=self.weight_dtype,
            )
        else:
            pipeline = AutoPipelineForInpainting.from_pretrained(
                self.model,
                vae=self.vae,
                text_encoder=self.text_encoder_one,
                text_encoder_2=self.text_encoder_two,
                tokenizer=self.tokenizer_one,
                tokenizer_2=self.tokenizer_two,
                unet=self.unet,
                safety_checker=None,
                torch_dtype=self.weight_dtype,
            )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for i, (p, img, m) in enumerate(zip(prompt, image, mask, strict=True)):
            generator = torch.Generator(device=self.device).manual_seed(i + seed)
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            mask_image = load_image(m) if isinstance(m, str) else m
            mask_image = mask_image.convert("L")
            image = pipeline(
                p,
                mask_image=mask_image,
                image=pil_img,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                generator=generator,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.

        """
        assert mode == "loss"
        num_batches = len(inputs["img"])

        latents = self._forward_vae(inputs["img"].to(self.weight_dtype), num_batches)
        masked_latents = self._forward_vae(
            inputs["masked_image"].to(self.weight_dtype), num_batches)

        mask = F.interpolate(inputs["mask"].to(self.weight_dtype),
                             size=(latents.shape[2], latents.shape[3]))

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

        latent_model_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

        if not self.pre_compute_text_embeddings:
            inputs["text_one"] = self.tokenizer_one(
                inputs["text"],
                max_length=self.tokenizer_one.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            inputs["text_two"] = self.tokenizer_two(
                inputs["text"],
                max_length=self.tokenizer_two.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                inputs["text_one"], inputs["text_two"])
        else:
            prompt_embeds = inputs["prompt_embeds"].to(self.weight_dtype)
            pooled_prompt_embeds = inputs["pooled_prompt_embeds"].to(self.weight_dtype)

        unet_added_conditions = {
            "time_ids": inputs["time_ids"].to(self.weight_dtype),
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
