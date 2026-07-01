import gradio as gr
from typing import List

import numpy as np
from PIL import Image

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel


# Load scheduler and models.
pretrained_model_path = 'preset/models/stable-diffusion-2-1-base'
seesr_model_path = 'preset/models/seesr'

scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(seesr_model_path, subfolder="unet")
controlnet = ControlNetModel.from_pretrained(seesr_model_path, subfolder="controlnet")

# Freeze models
vae.requires_grad_(False)
unet.requires_grad_(False)
controlnet.requires_grad_(False)

if is_xformers_available():
    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
else:
    raise ValueError("xformers is not available. Make sure it is installed correctly")

# Get the validation pipeline
validation_pipeline = StableDiffusionControlNetPipeline(
    vae=vae, unet=unet, controlnet=controlnet, scheduler=scheduler,
)

validation_pipeline._init_tiled_vae(encoder_tile_size=1024,
                                    decoder_tile_size=224)
weight_dtype = torch.float16
device = "cuda"


def pad_image_to_multiple(image: Image.Image, multiple: int = 8):
    width, height = image.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    if pad_width == 0 and pad_height == 0:
        return image, (0, 0)

    image_array = np.asarray(image)
    padded_array = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return Image.fromarray(padded_array), (pad_width, pad_height)

# Move models to gpu and cast to weight_dtype
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)


@torch.no_grad()
def process(
    input_image: Image.Image,
    num_inference_steps: int,
    seed: int,
    latent_tiled_size: int,
    latent_tiled_overlap: int,
    sample_times: int
    ) -> List[np.ndarray]:
    # with torch.no_grad():
    set_seed(seed)
    generator = torch.Generator(device=device)

    ori_width, ori_height = input_image.size
    input_image, padding = pad_image_to_multiple(input_image, multiple=8)
    pad_width, pad_height = padding

    width, height = input_image.size

    images = []
    for _ in range(sample_times):
        try:
            with torch.autocast("cuda"):
                image = validation_pipeline(
                    input_image, num_inference_steps=num_inference_steps, generator=generator,
                    height=height, width=width,
                    conditioning_scale=1,
                    start_point='lr', start_steps=999,
                    latent_tiled_size=latent_tiled_size, latent_tiled_overlap=latent_tiled_overlap
                ).images[0]

            if True:  # alpha<1.0:
                image = wavelet_color_fix(image, input_image)

            if pad_width or pad_height:
                image = image.crop((0, 0, ori_width, ori_height))
        except Exception as e:
            print(e)
            image = Image.new(mode="RGB", size=(512, 512))
        images.append(np.array(image))
    return images


#
MARKDOWN = \
"""
## SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution

[GitHub](https://github.com/cswry/SeeSR) | [Paper](https://arxiv.org/abs/2311.16518)

If SeeSR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
                sample_times = gr.Slider(label="Sample Times", minimum=1, maximum=10, step=1, value=1)
                latent_tiled_size = gr.Slider(label="Diffusion Tile Size", minimum=128, maximum=480, value=320, step=1)
                latent_tiled_overlap = gr.Slider(label="Diffusion Tile Overlap", minimum=4, maximum=16, value=4, step=1)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=2, height="auto")

    inputs = [
        input_image,
        num_inference_steps,
        seed,
        latent_tiled_size,
        latent_tiled_overlap,
        sample_times,
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch()
