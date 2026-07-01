# Prediction interface for Cog
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import time
from typing import List

import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

MODEL_URL = "https://weights.replicate.delivery/default/stabilityai/sd-2-1-base.tar"
DEVICE = "cuda"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def pad_image_to_multiple(image: Image.Image, multiple: int = 8):
    width, height = image.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    if pad_width == 0 and pad_height == 0:
        return image, (0, 0)

    image_array = np.asarray(image)
    padded_array = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return Image.fromarray(padded_array), (pad_width, pad_height)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the deblurring pipeline once for repeated predictions."""
        pretrained_model_path = "preset/models/stable-diffusion-2-1-base"
        seesr_model_path = "preset/models/seesr"

        if not os.path.exists(pretrained_model_path):
            download_weights(MODEL_URL, pretrained_model_path)

        scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(seesr_model_path, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(seesr_model_path, subfolder="controlnet")

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        controlnet.requires_grad_(False)

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        pipeline._init_tiled_vae(encoder_tile_size=1024, decoder_tile_size=224)

        weight_dtype = torch.float16
        vae.to(DEVICE, dtype=weight_dtype)
        unet.to(DEVICE, dtype=weight_dtype)
        controlnet.to(DEVICE, dtype=weight_dtype)

        self.pipeline = pipeline

    @torch.inference_mode()
    def process(
        self,
        input_image: Image.Image,
        num_inference_steps: int,
        conditioning_scale: float,
        seed: int,
        latent_tiled_size: int,
        latent_tiled_overlap: int,
        sample_times: int,
        align_method: str,
    ) -> List[np.ndarray]:
        set_seed(seed)
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)

        ori_width, ori_height = input_image.size
        input_image, padding = pad_image_to_multiple(input_image, multiple=8)
        pad_width, pad_height = padding
        width, height = input_image.size

        images = []
        for _ in range(sample_times):
            try:
                with torch.autocast("cuda"):
                    image = self.pipeline(
                        input_image,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        height=height,
                        width=width,
                        conditioning_scale=conditioning_scale,
                        start_point="lr",
                        start_steps=999,
                        latent_tiled_size=latent_tiled_size,
                        latent_tiled_overlap=latent_tiled_overlap,
                    ).images[0]

                if align_method == "wavelet":
                    image = wavelet_color_fix(image, input_image)
                elif align_method == "adain":
                    image = adain_color_fix(image, input_image)

                if pad_width or pad_height:
                    image = image.crop((0, 0, ori_width, ori_height))
            except Exception as exc:
                print(exc)
                image = Image.new(mode="RGB", size=(ori_width, ori_height))
            images.append(np.array(image))
        return images

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input blurry image"),
        num_inference_steps: int = Input(description="Number of inference steps", default=50, ge=1, le=100),
        sample_times: int = Input(description="Number of samples to generate", default=1, ge=1, le=10),
        latent_tiled_size: int = Input(description="Latent tile size", default=96, ge=32, le=480),
        latent_tiled_overlap: int = Input(description="Latent tile overlap", default=4, ge=0, le=32),
        conditioning_scale: float = Input(description="ControlNet conditioning scale", default=1.0, ge=0.0, le=2.0),
        align_method: str = Input(description="Color alignment method: adain, wavelet, or nofix", default="adain"),
        seed: int = Input(description="Seed", default=231, ge=0, le=2147483647),
    ) -> List[Path]:
        pil_image = Image.open(image).convert("RGB")
        imgs = self.process(
            pil_image,
            num_inference_steps,
            conditioning_scale,
            seed,
            latent_tiled_size,
            latent_tiled_overlap,
            sample_times,
            align_method,
        )

        output_dir = "/tmp/output"
        os.makedirs(output_dir, exist_ok=True)
        for existing_name in os.listdir(output_dir):
            existing_path = os.path.join(output_dir, existing_name)
            if os.path.isfile(existing_path):
                os.remove(existing_path)

        output_paths = []
        for i, img in enumerate(imgs):
            output_path = f"{output_dir}/{i}.png"
            Image.fromarray(img).save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
