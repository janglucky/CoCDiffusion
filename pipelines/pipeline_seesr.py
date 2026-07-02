# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    BaseOutput,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)

from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from models.controlnet import ControlNetModel as LocalControlNetModel

from schedulers.coc_blur_scheduler import CoCBlurScheduler
from schedulers.coc_endpoint_scheduler import CoCEndpointScheduler
from schedulers.coc_image_latent_scheduler import CoCImageLatentScheduler
from schedulers.gaussian_blur_scheduler import GaussianBlurScheduler
from schedulers.paired_endpoint_scheduler import PairedEndpointScheduler
from utils.vaehook import VAEHook, perfcount


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SINGLE_CONTROLNET_TYPES = (ControlNetModel, LocalControlNetModel)


@dataclass
class DeblurPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]] = None


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> # download an image
        >>> blurry_image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization.
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(blurry_image, num_inference_steps=20, generator=generator).images[0]
        ```
"""


class StableDiffusionControlNetPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-conditioned deblurring using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        if encoder_tile_size is not None and encoder_tile_size > 0:
            self.vae.encoder.forward = VAEHook(
                encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        else:
            self.vae.encoder.forward = self.vae.encoder.original_forward

        if decoder_tile_size is not None and decoder_tile_size > 0:
            self.vae.decoder.forward = VAEHook(
                decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        else:
            self.vae.decoder.forward = self.vae.decoder.original_forward

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
            vae, and controlnet have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _infer_batch_size(self, image):
        if isinstance(image, list):
            return len(image)
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return image.shape[0]
        return 1

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        #extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        image,
        height,
        width,
        callback_steps,
        controlnet_conditioning_scale=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, SINGLE_CONTROLNET_TYPES)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, SINGLE_CONTROLNET_TYPES)
        ):
            self.check_image(image)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    "For multiple controlnets: `image` must have the same length as the number of controlnets."
                )

            for image_ in image:
                self.check_image(image_)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, SINGLE_CONTROLNET_TYPES)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, SINGLE_CONTROLNET_TYPES)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

    def check_image(self, image):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

        if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_input,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    #image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)#.flip(1)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is already aligned with input batch size
            repeat_by = num_images_per_input

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_depth(
        self,
        depth,
        width,
        height,
        batch_size,
        num_images_per_input,
        device,
        dtype,
    ):
        if depth is None:
            return None

        if not isinstance(depth, torch.Tensor):
            if isinstance(depth, PIL.Image.Image):
                depth = [depth]

            if isinstance(depth[0], PIL.Image.Image):
                depths = []
                for depth_ in depth:
                    depth_ = depth_.convert("L")
                    depth_array = np.array(depth_, dtype=np.float32) / 255.0
                    depths.append(depth_array[None, None, :, :])
                depth = torch.from_numpy(np.concatenate(depths, axis=0))
            elif isinstance(depth[0], torch.Tensor):
                depth = torch.cat(depth, dim=0)

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] != 1:
            depth = depth[:, :1]

        if depth.shape[-2:] != (height, width):
            depth = F.interpolate(depth.float(), size=(height, width), mode="bilinear", align_corners=False)

        depth_batch_size = depth.shape[0]
        repeat_by = batch_size if depth_batch_size == 1 else num_images_per_input
        depth = depth.repeat_interleave(repeat_by, dim=0)
        return depth.to(device=device, dtype=dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            #latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
            #offset_noise = torch.randn(batch_size, num_channels_latents, 1, 1, device=device)
            #latents = latents + 0.1 * offset_noise
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        if isinstance(self.controlnet, ControlNetModel):
            super().save_pretrained(save_directory, safe_serialization, variant)
        else:
            raise NotImplementedError("Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.")
        
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

    @perfcount
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        depth: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_input: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        image_sr = None,
        start_steps = 999,
        start_point = 'noise',
        latent_tiled_size=320,
        latent_tiled_overlap=4,
        diffusion_process="gaussian",
        coc_focus_depth=0.7,
        coc_focus_width=0.0,
        coc_focus_depth_min=0.1,
        coc_focus_depth_max=0.9,
        coc_focus_width_min=0.0,
        coc_focus_width_max=0.12,
        coc_global_blur_min=0.0,
        coc_global_blur_max=1.0,
        coc_max_radius=2.5,
        coc_gamma=1.5,
        coc_schedule_power=3.0,
        coc_global_blur_at_max=0.0,
        coc_depth_blur_strength=1.0,
        coc_inference_start="encoded_input",
        start_blur_sigma=8.0,
        start_blur_kernel_size=None,
        update_blend=1.0,
        args=None
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            num_images_per_input (`int`, *optional*, defaults to 1):
                The number of images to generate per input image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. If not provided, a latents tensor will ge generated by sampling using the supplied random
                `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DeblurPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                Kept for API compatibility.

        Examples:

        Returns:
            [`DeblurPipelineOutput`] or `tuple`:
            [`DeblurPipelineOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is a list with the generated images.
        """
        if image is None:
            raise ValueError("`image` must be provided.")
        if diffusion_process not in ("gaussian", "coc_blur", "paired_endpoint", "coc_endpoint", "coc_image_latent"):
            raise ValueError(
                "`diffusion_process` must be 'gaussian', 'coc_blur', 'paired_endpoint', "
                "'coc_endpoint', or 'coc_image_latent'."
            )
        if coc_inference_start not in ("latent_max_blur", "encoded_input", "gaussian_blur"):
            raise ValueError(
                "`coc_inference_start` must be one of 'latent_max_blur', 'encoded_input', or 'gaussian_blur'."
            )

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps, conditioning_scale)

        # 2. Define call parameters
        batch_size = self._infer_batch_size(image)

        device = self._execution_device
        do_classifier_free_guidance = False

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        """
        if isinstance(controlnet, MultiControlNetModel) and isinstance(conditioning_scale, float):
            conditioning_scale = [conditioning_scale] * len(controlnet.nets)
        
        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        
        guess_mode = guess_mode or global_pool_conditions
        """

        # 4. Prepare image
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_input,
            num_images_per_input=num_images_per_input,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        depth = self.prepare_depth(
            depth=depth,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_input,
            num_images_per_input=num_images_per_input,
            device=device,
            dtype=controlnet.dtype,
        )

        use_timestep_conditioning = getattr(self.unet.config, "use_timestep_conditioning", True)

        # 5. Prepare timesteps. CoC still uses a multi-step blur/deblur schedule, but
        # no-time models receive None instead of the explicit timestep embedding.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        coc_blur_scheduler = None
        paired_endpoint_scheduler = None
        coc_endpoint_scheduler = None
        coc_image_latent_scheduler = None
        if diffusion_process == "coc_blur":
            coc_blur_scheduler = CoCBlurScheduler(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                focus_depth=coc_focus_depth,
                focus_width=coc_focus_width,
                max_radius=coc_max_radius,
                gamma=coc_gamma,
                schedule_power=coc_schedule_power,
                global_blur_at_max=coc_global_blur_at_max,
                depth_blur_strength=coc_depth_blur_strength,
                focus_depth_min=coc_focus_depth_min,
                focus_depth_max=coc_focus_depth_max,
                focus_width_min=coc_focus_width_min,
                focus_width_max=coc_focus_width_max,
                global_blur_min=coc_global_blur_min,
                global_blur_max=coc_global_blur_max,
            )
        elif diffusion_process == "paired_endpoint":
            paired_endpoint_scheduler = PairedEndpointScheduler(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                schedule_power=coc_schedule_power,
            )
        elif diffusion_process == "coc_endpoint":
            coc_endpoint_scheduler = CoCEndpointScheduler(
                CoCBlurScheduler(
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    focus_depth=coc_focus_depth,
                    focus_width=coc_focus_width,
                    max_radius=coc_max_radius,
                    gamma=coc_gamma,
                    schedule_power=coc_schedule_power,
                    global_blur_at_max=coc_global_blur_at_max,
                    depth_blur_strength=coc_depth_blur_strength,
                    focus_depth_min=coc_focus_depth_min,
                    focus_depth_max=coc_focus_depth_max,
                    focus_width_min=coc_focus_width_min,
                    focus_width_max=coc_focus_width_max,
                    global_blur_min=coc_global_blur_min,
                    global_blur_max=coc_global_blur_max,
                )
            )
        elif diffusion_process == "coc_image_latent":
            coc_image_latent_scheduler = CoCImageLatentScheduler(
                CoCBlurScheduler(
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    focus_depth=coc_focus_depth,
                    focus_width=coc_focus_width,
                    max_radius=coc_max_radius,
                    gamma=coc_gamma,
                    schedule_power=coc_schedule_power,
                    global_blur_at_max=coc_global_blur_at_max,
                    depth_blur_strength=coc_depth_blur_strength,
                    focus_depth_min=coc_focus_depth_min,
                    focus_depth_max=coc_focus_depth_max,
                    focus_width_min=coc_focus_width_min,
                    focus_width_max=coc_focus_width_max,
                    global_blur_min=coc_global_blur_min,
                    global_blur_max=coc_global_blur_max,
                )
            )

        vae_dtype = next(self.vae.parameters()).dtype

        def encode_image_to_latents(image_tensor):
            image_tensor = image_tensor.to(device=device, dtype=vae_dtype)
            encoded_latents = self.vae.encode(image_tensor).latent_dist.sample()
            return (encoded_latents * self.vae.config.scaling_factor).to(dtype=controlnet.dtype)

        def decode_latents_to_image(latents_tensor):
            latents_tensor = (latents_tensor / self.vae.config.scaling_factor).to(dtype=vae_dtype)
            decoded_image = self.vae.decode(latents_tensor, return_dict=False)[0]
            return decoded_image.to(dtype=controlnet.dtype)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        paired_degraded_endpoint_latents = None
        coc_endpoint_degraded_latents = None
        if diffusion_process == "paired_endpoint":
            paired_degraded_endpoint_latents = self.vae.encode(image * 2 - 1).latent_dist.sample()
            paired_degraded_endpoint_latents = paired_degraded_endpoint_latents * self.vae.config.scaling_factor
            latents = paired_degraded_endpoint_latents
        elif diffusion_process == "coc_endpoint":
            coc_endpoint_degraded_latents = self.vae.encode(image * 2 - 1).latent_dist.sample()
            coc_endpoint_degraded_latents = coc_endpoint_degraded_latents * self.vae.config.scaling_factor
            latents = coc_endpoint_degraded_latents
        elif diffusion_process == "coc_image_latent":
            latents = encode_image_to_latents(image * 2 - 1)
        elif diffusion_process == "coc_blur":
            if coc_inference_start == "gaussian_blur":
                blur_scheduler = GaussianBlurScheduler(
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    max_sigma=start_blur_sigma,
                    kernel_size=start_blur_kernel_size,
                    schedule_power=1.0,
                )
                last_timestep = torch.full(
                    (image.shape[0],),
                    self.scheduler.config.num_train_timesteps - 1,
                    device=image.device,
                    dtype=torch.long,
                )
                image_for_start = blur_scheduler.add_blur(image, last_timestep)
                latents = self.vae.encode(image_for_start * 2 - 1).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            else:
                latents = self.vae.encode(image * 2 - 1).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                if coc_inference_start == "latent_max_blur":
                    max_blur_timestep = torch.full(
                        (latents.shape[0],),
                        self.scheduler.config.num_train_timesteps - 1,
                        device=latents.device,
                        dtype=torch.long,
                    )
                    full_blur_depth = torch.zeros(
                        (latents.shape[0], 1, latents.shape[-2], latents.shape[-1]),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    latents = coc_blur_scheduler.add_blur(
                        latents,
                        full_blur_depth,
                        max_blur_timestep,
                        focus_depth=0.5,
                        focus_width=0.0,
                        global_blur_floor=1.0,
                    )
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_input,
                num_channels_latents,
                height,
                width,
                controlnet.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare the start point
            if start_point == 'noise':
                latents = latents
            elif start_point == 'lr': # LRE Strategy
                latents_condition_image = self.vae.encode(image*2-1).latent_dist.sample()
                latents_condition_image = latents_condition_image * self.vae.config.scaling_factor
                start_steps_tensor = torch.randint(start_steps, start_steps+1, (latents.shape[0],), device=latents.device)
                start_steps_tensor = start_steps_tensor.long()
                latents = self.scheduler.add_noise(latents_condition_image[0:1, ...], latents, start_steps_tensor)
    
        coc_sample_focus_depth = None
        coc_sample_focus_width = None
        coc_sample_global_blur_floor = None
        if diffusion_process in ("coc_blur", "coc_endpoint", "coc_image_latent") and depth is not None:
            if diffusion_process == "coc_blur" and coc_inference_start == "latent_max_blur":
                coc_sample_focus_depth = torch.full(
                    (latents.shape[0],), 0.5, device=latents.device, dtype=latents.dtype
                )
                coc_sample_focus_width = torch.zeros(
                    (latents.shape[0],), device=latents.device, dtype=latents.dtype
                )
                coc_sample_global_blur_floor = torch.ones(
                    (latents.shape[0],), device=latents.device, dtype=latents.dtype
                )
            else:
                if diffusion_process == "coc_blur":
                    dof_scheduler = coc_blur_scheduler
                elif diffusion_process == "coc_endpoint":
                    dof_scheduler = coc_endpoint_scheduler
                else:
                    dof_scheduler = coc_image_latent_scheduler
                (
                    coc_sample_focus_depth,
                    coc_sample_focus_width,
                    coc_sample_global_blur_floor,
                ) = dof_scheduler.sample_dof_params(latents)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        progress_total = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=progress_total) as progress_bar:
            
            _, _, h, w = latents.size()
            tile_size, tile_overlap = (latent_tiled_size, latent_tiled_overlap) if args is not None else (256, 8)
            use_latent_tiling = tile_size is not None and tile_size > 0 and h * w > tile_size * tile_size
            if tile_size is None or tile_size <= 0:
                print("[Tiled Latent]: disabled, using full latent inference.")
            elif not use_latent_tiling:
                print("[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            else:
                print(f"[Tiled Latent]: the input size is {image.shape[-2]}x{image.shape[-1]}, need to tiled")

            for i, t in enumerate(timesteps):
                # pass, if the timestep is larger than start_steps
                if t > start_steps:
                    print(f'pass {t} steps.')
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if diffusion_process == "gaussian":
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                model_timestep = t if use_timestep_conditioning else None

                # controlnet(s) inference
                controlnet_latent_model_input = latent_model_input

                if not use_latent_tiling:
                    down_block_res_samples, mid_block_res_sample = [None]*10, None
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        controlnet_latent_model_input,
                        model_timestep,
                        controlnet_cond=image,
                        conditioning_scale=conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )


                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        model_timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                else:
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
                    tile_size = min(tile_size, min(h, w))
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

                    grid_rows = 0
                    cur_x = 0
                    while cur_x < latent_model_input.size(-1):
                        cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                        grid_rows += 1

                    grid_cols = 0
                    cur_y = 0
                    while cur_y < latent_model_input.size(-2):
                        cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                        grid_cols += 1

                    input_list = []
                    cond_list = []
                    img_list = []
                    noise_preds = []
                    for row in range(grid_rows):
                        noise_preds_row = []
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size

                            # input tile dimensions
                            input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            input_list.append(input_tile)
                            cond_tile = controlnet_latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            cond_list.append(cond_tile)
                            img_tile = image[:, :, input_start_y*8:input_end_y*8, input_start_x*8:input_end_x*8]
                            img_list.append(img_tile)

                            if len(input_list) == batch_size or col == grid_cols-1:
                                input_list_t = torch.cat(input_list, dim=0)
                                cond_list_t = torch.cat(cond_list, dim=0)
                                img_list_t = torch.cat(img_list, dim=0)
                                #print(input_list_t.shape, cond_list_t.shape, img_list_t.shape, fg_mask_list_t.shape)

                                down_block_res_samples, mid_block_res_sample = self.controlnet(
                                    cond_list_t,
                                    model_timestep,
                                    controlnet_cond=img_list_t,
                                    conditioning_scale=conditioning_scale,
                                    guess_mode=guess_mode,
                                    return_dict=False,
                                )

                                if guess_mode and do_classifier_free_guidance:
                                    # Infered ControlNet only for the conditional batch.
                                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                                    # add 0 to the unconditional batch to keep it unchanged.
                                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                                # predict the noise residual
                                model_out = self.unet(
                                    input_list_t,
                                    model_timestep,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample,
                                    return_dict=False,
                                )[0]

                                #for sample_i in range(model_out.size(0)):
                                #    noise_preds_row.append(model_out[sample_i].unsqueeze(0))
                                input_list = []
                                cond_list = []
                                img_list = []

                            noise_preds.append(model_out)

                    # Stitch noise predictions for all tiles
                    noise_pred = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    contributors = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    # Add each tile contribution to overall latents
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size
    
                            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                    # Average overlapping areas with more than 1 contributor
                    noise_pred /= contributors
                if diffusion_process == "gaussian":
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                elif diffusion_process == "coc_blur":
                    if depth is not None:
                        latents = coc_blur_scheduler.step(
                            predicted_clean=noise_pred,
                            sample=latents,
                            depth=depth,
                            timestep=t,
                            inference_timesteps=timesteps,
                            focus_depth=coc_sample_focus_depth,
                            focus_width=coc_sample_focus_width,
                            global_blur_floor=coc_sample_global_blur_floor,
                        )
                    else:
                        index = (timesteps == t).nonzero(as_tuple=False)
                        if len(index) == 0 or int(index[0].item()) + 1 >= len(timesteps):
                            prev_timestep = torch.zeros_like(t)
                        else:
                            prev_timestep = timesteps[int(index[0].item()) + 1]
                        denom = max(self.scheduler.config.num_train_timesteps - 1, 1)
                        current_strength = float(t.item()) / denom
                        prev_strength = float(prev_timestep.item()) / denom
                        step_fraction = (current_strength - prev_strength) / max(current_strength, 1e-6)
                        step_fraction = min(max(step_fraction * float(update_blend), 0.0), 1.0)
                        latents = (1.0 - step_fraction) * latents + step_fraction * noise_pred
                elif diffusion_process == "paired_endpoint":
                    latents = paired_endpoint_scheduler.step(
                        predicted_clean=noise_pred,
                        sample=latents,
                        degraded_endpoint=paired_degraded_endpoint_latents,
                        timestep=t,
                        inference_timesteps=timesteps,
                    )
                elif diffusion_process == "coc_endpoint":
                    latents = coc_endpoint_scheduler.step(
                        predicted_clean=noise_pred,
                        sample=latents,
                        degraded_endpoint=coc_endpoint_degraded_latents,
                        depth=depth,
                        timestep=t,
                        inference_timesteps=timesteps,
                        focus_depth=coc_sample_focus_depth,
                        focus_width=coc_sample_focus_width,
                        global_blur_floor=coc_sample_global_blur_floor,
                    )
                else:
                    latents = coc_image_latent_scheduler.step(
                        predicted_clean=noise_pred,
                        sample=latents,
                        depth=depth,
                        timestep=t,
                        inference_timesteps=timesteps,
                        decode_latents_to_image=decode_latents_to_image,
                        encode_image_to_latents=encode_image_to_latents,
                        focus_depth=coc_sample_focus_depth,
                        focus_width=coc_sample_focus_width,
                        global_blur_floor=coc_sample_global_blur_floor,
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        has_nsfw_concept = None
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]#.flip(1)
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return DeblurPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
