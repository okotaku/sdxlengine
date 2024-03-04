from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from peft import get_peft_model
from torch import nn

from diffengine.models.editors.stable_diffusion_xl.data_preprocessor import (
    DataPreprocessor,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import TimeSteps, WhiteNoise

weight_dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class StableDiffusionXL(BaseModel):
    """Stable Diffusion XL.

    <https://huggingface.co/papers/2307.01952>`_

    Args:
    ----
        tokenizer_one (dict): Config of tokenizer one.
        tokenizer_two (dict): Config of tokenizer two.
        scheduler (dict): Config of scheduler.
        text_encoder_one (dict): Config of text encoder one.
        text_encoder_two (dict): Config of text encoder two.
        vae (dict): Config of vae.
        unet (dict): Config of unet.
        model (str): pretrained model name of stable diffusion.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        unet_lora_config (dict, optional): The LoRA config dict for Unet.
            example. dict(type=LoraConfig, r=4). Refer to the PEFT for more
            details. https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder.example. dict(type=LoraConfig, r=4). Refer to the PEFT
            for more details. https://github.com/huggingface/peft
            Defaults to None.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler will be used. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`DataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        vae_batch_size (int): The batch size of vae. Defaults to 8.
        weight_dtype (str): The weight dtype. Choose from "fp32", "fp16" or
            "bf16".  Defaults to 'fp32'.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        pre_compute_text_embeddings (bool): Whether or not to pre-compute text
            embeddings to save memory. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.

    """

    def __init__(  # noqa: PLR0913,C901,PLR0912,PLR0915
        self,
        tokenizer_one: dict,
        tokenizer_two: dict,
        scheduler: dict,
        text_encoder_one: dict,
        text_encoder_two: dict,
        vae: dict,
        unet: dict,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        vae_batch_size: int = 8,
        weight_dtype: str = "fp32",
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": DataPreprocessor}
        if loss is None:
            loss = {}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            unet_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Unet and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and unet_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Unet, "
                "you should set text_encoder_lora_config."
            )
        if pre_compute_text_embeddings:
            assert not finetune_text_encoder

        self.model = model
        self.unet_lora_config = deepcopy(unet_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers
        self.vae_batch_size = vae_batch_size
        self.weight_dtype = weight_dtype_dict[weight_dtype]

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": L2Loss, "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        if not self.pre_compute_text_embeddings:
            self.tokenizer_one = MODELS.build(
                tokenizer_one,
                default_args={"pretrained_model_name_or_path": model})
            self.tokenizer_two = MODELS.build(
                tokenizer_two,
                default_args={"pretrained_model_name_or_path": model})

            self.text_encoder_one = MODELS.build(
                text_encoder_one,
                default_args={"pretrained_model_name_or_path": model})
            self.text_encoder_two = MODELS.build(
                text_encoder_two,
                default_args={"pretrained_model_name_or_path": model})

        self.scheduler = MODELS.build(
            scheduler,
            default_args={"pretrained_model_name_or_path": model})

        self.vae = MODELS.build(
            vae,
            default_args={"pretrained_model_name_or_path": model})
        self.unet = MODELS.build(
            unet,
            default_args={"pretrained_model_name_or_path": model})
        self.noise_generator = MODELS.build(
            noise_generator,
            default_args={"type": WhiteNoise})
        self.timesteps_generator = MODELS.build(
            timesteps_generator,
            default_args={"type": TimeSteps})

        if hasattr(self.vae.config, "latents_mean",
                   ) and self.vae.config.latents_mean is not None:
            self.edm_style = True
            self.register_buffer(
                "latents_mean",
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1))
            self.register_buffer(
                "latents_std",
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1))
            self.register_buffer("sigmas", self.scheduler.sigmas)
        else:
            self.edm_style = False

        self.prepare_model()
        self.set_lora()
        self.set_xformers()

        if self.weight_dtype != torch.float32:
            self.to(self.weight_dtype)

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = MODELS.build(
                self.text_encoder_lora_config)
            self.text_encoder_one = get_peft_model(
                self.text_encoder_one, text_encoder_lora_config)
            self.text_encoder_one.print_trainable_parameters()
            self.text_encoder_two = get_peft_model(
                self.text_encoder_two, text_encoder_lora_config)
            self.text_encoder_two.print_trainable_parameters()
        if self.unet_lora_config is not None:
            unet_lora_config = MODELS.build(
                self.unet_lora_config)
            self.unet = get_peft_model(self.unet, unet_lora_config)
            self.unet.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
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

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.

        """
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
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
        if self.pre_compute_text_embeddings:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                unet=self.unet,
                safety_checker=None,
                torch_dtype=self.weight_dtype,
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
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
        for i, p in enumerate(prompt):
            generator = torch.Generator(device=self.device).manual_seed(i + seed)
            image = pipeline(
                p,
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

    def val_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Val step."""
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Test step."""
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def encode_prompt(
        self,
        text_one: torch.Tensor,
        text_two: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt.

        Args:
        ----
            text_one (torch.Tensor): Token ids from tokenizer one.
            text_two (torch.Tensor): Token ids from tokenizer two.

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: Prompt embeddings

        """
        prompt_embeds_list = []

        text_encoders = [self.text_encoder_one, self.text_encoder_two]
        texts = [text_one, text_two]
        for text_encoder, text in zip(text_encoders, texts, strict=True):

            prompt_embeds = text_encoder(
                text,
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the
            # final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def loss(self,
             model_pred: torch.Tensor,
             noise: torch.Tensor,
             latents: torch.Tensor,
             timesteps: torch.Tensor,
             noisy_model_input: torch.Tensor,
             sigmas: torch.Tensor | None = None,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        if self.edm_style:
            model_pred = self.scheduler.precondition_outputs(
                noisy_model_input, model_pred, sigmas)

        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        if self.edm_style:
            gt = latents
        elif self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict

    def _preprocess_model_input(self,
                                latents: torch.Tensor,
                                noise: torch.Tensor,
                                timesteps: torch.Tensor) -> torch.Tensor:
        """Preprocess model input."""
        if self.input_perturbation_gamma > 0:
            input_noise = noise + self.input_perturbation_gamma * torch.randn_like(
                noise)
        else:
            input_noise = noise
        noisy_model_input = self.scheduler.add_noise(
            latents, input_noise, timesteps)
        if self.edm_style:
            sigmas =self._get_sigmas(timesteps)
            inp_noisy_latents = self.scheduler.precondition_inputs(
                noisy_model_input, sigmas)
        else:
            inp_noisy_latents = noisy_model_input
            sigmas = None
        return noisy_model_input, inp_noisy_latents, sigmas

    def _forward_vae(self, img: torch.Tensor, num_batches: int,
                     ) -> torch.Tensor:
        """Forward vae."""
        latents = [
            self.vae.encode(
                img[i : i + self.vae_batch_size],
            ).latent_dist.sample() for i in range(
                0, num_batches, self.vae_batch_size)
        ]
        latents = torch.cat(latents, dim=0)
        if hasattr(self, "latents_mean"):
            return (
                latents - self.latents_mean
            ) * self.vae.config.scaling_factor / self.latents_std
        return latents * self.vae.config.scaling_factor

    def _get_sigmas(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sigmas."""
        step_indices = [(
            self.scheduler.timesteps.to(self.device) == t
            ).nonzero().item() for t in timesteps]
        sigma = self.sigmas[step_indices].flatten()
        return sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

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

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

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
            inp_noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
