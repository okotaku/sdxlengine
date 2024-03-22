# Copied from https://github.com/siliconflow/onediff/tree/18295b04a0168878c89b746ed892a577961a4726/onediff_diffusers_extensions
import os
import warnings

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import deprecate
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils.log_utils import logger
from onediff.infer_compiler.with_oneflow_compile import DeployableModule
from PIL import Image


def patch_image_prcessor(processor):
    if type(processor) is VaeImageProcessor:
        processor.postprocess = postprocess.__get__(processor)
        processor.pt_to_numpy = pt_to_numpy.__get__(processor)
        processor.pt_to_pil = pt_to_pil.__get__(processor)
    else:
        warnings.warn(
            f"Image processor {type(processor)} is not supported for patching",
        )


def postprocess(
    self,
    image: torch.FloatTensor,
    output_type: str = "pil",
    do_denormalize: list[bool] | None = None,
):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor",
        )
    if output_type not in ["latent", "pt", "np", "pil"]:
        deprecation_message = (
            f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
            "`pil`, `np`, `pt`, `latent`"
        )
        deprecate(
            "Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False,
        )
        output_type = "np"

    if output_type == "latent":
        return image

    if do_denormalize is None:
        do_denormalize = [self.config.do_normalize] * image.shape[0]

    image = torch.stack(
        [
            self.denormalize(image[i]) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ],
    )

    if output_type == "pt":
        return image

    if output_type == "pil":
        return self.pt_to_pil(image)

    image = self.pt_to_numpy(image)

    if output_type == "np":
        return image

    # if output_type == "pil":
    #     return self.numpy_to_pil(image)


@torch.jit.script
def _pt_to_numpy_pre(images):
    return images.permute(0, 2, 3, 1).contiguous().float().cpu()


@staticmethod  # type: ignore[misc]
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    # images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    # return images
    return _pt_to_numpy_pre(images).numpy()


@torch.jit.script
def _pt_to_pil_pre(images):
    return (
        images.permute(0, 2, 3, 1)
        .contiguous()
        .float()
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


@staticmethod  # type: ignore[misc]
def pt_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    # images = (images * 255).round().astype("uint8")
    images = _pt_to_pil_pre(images).numpy()
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def _recursive_getattr(obj, attr, default=None):
    attrs = attr.split(".")
    for attr in attrs:
        if not hasattr(obj, attr):
            return default
        obj = getattr(obj, attr, default)
    return obj


def _recursive_setattr(obj, attr, value):
    attrs = attr.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


_PARTS = [
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "unet",
    "controlnet",
    "fast_unet",  # for deepcache
    "vae.decoder",
    "vae.encoder",
]

def _filter_parts(ignores=()):
    filtered_parts = []
    for part in _PARTS:
        skip = False
        for ignore in ignores:
            if part == ignore or part.startswith(ignore + "."):
                skip = True
                break
        if not skip:
            filtered_parts.append(part)

    return filtered_parts

def compile_pipe(
    pipe, *, ignores=(),
):
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None:
            logger.info(f"Compiling {part}")
            _recursive_setattr(pipe, part, oneflow_compile(obj))

    if "image_processor" not in ignores:
        logger.info("Patching image_processor")

        patch_image_prcessor(pipe.image_processor)

    return pipe


def save_pipe(
    pipe, dst_dir="cached_pipe", *, ignores=(), overwrite=True,
):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if (
            obj is not None
            and isinstance(obj, DeployableModule)
            and obj._deployable_module_dpl_graph is not None
            and obj.get_graph().is_compiled
        ):
            if not overwrite and os.path.isfile(os.path.join(dst_dir, part)):
                logger.info(f"Compiled graph already exists for {part}, not overwriting it.")
                continue
            logger.info(f"Saving {part}")
            obj.save_graph(os.path.join(dst_dir, part))


def load_pipe(
    pipe, src_dir="cached_pipe", *, ignores=(),
):
    if not os.path.exists(src_dir):
        return
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None and os.path.exists(os.path.join(src_dir, part)):
            logger.info(f"Loading {part}")
            obj.load_graph(os.path.join(src_dir, part))

    if "image_processor" not in ignores:
        logger.info("Patching image_processor")

        patch_image_prcessor(pipe.image_processor)
