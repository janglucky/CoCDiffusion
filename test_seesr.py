'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import numpy as np
from PIL import Image

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

logger = get_logger(__name__, log_level="INFO")


def pad_image_to_multiple(image, multiple=8):
    width, height = image.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    if pad_width == 0 and pad_height == 0:
        return image, (0, 0)

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        pad_widths = ((0, pad_height), (0, pad_width))
    else:
        pad_widths = ((0, pad_height), (0, pad_width), (0, 0))
    padded_array = np.pad(image_array, pad_widths, mode="edge")
    return Image.fromarray(padded_array), (pad_width, pad_height)


def infer_depth_path(image_path):
    image_path = os.path.abspath(image_path)
    parent = os.path.dirname(image_path)
    if os.path.basename(parent) == "source":
        candidate = os.path.join(os.path.dirname(parent), "depth", os.path.basename(image_path))
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_depth_path(depth_path, image_path):
    if depth_path is None:
        return infer_depth_path(image_path)
    if os.path.isdir(depth_path):
        candidate = os.path.join(depth_path, os.path.basename(image_path))
        if os.path.exists(candidate):
            return candidate
        stem = os.path.splitext(os.path.basename(image_path))[0]
        matches = sorted(glob.glob(os.path.join(depth_path, f"{stem}.*")))
        return matches[0] if matches else None
    return depth_path


def resolve_timestep_conditioning(args):
    if args.timestep_conditioning == "on":
        return True
    if args.timestep_conditioning == "off":
        return False
    return args.diffusion_process in ("gaussian", "coc_image_latent")


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler and models.
    use_timestep_conditioning = resolve_timestep_conditioning(args)

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.seesr_model_path,
        subfolder="unet",
        use_timestep_conditioning=use_timestep_conditioning,
    )
    controlnet = ControlNetModel.from_pretrained(
        args.seesr_model_path,
        subfolder="controlnet",
        use_timestep_conditioning=use_timestep_conditioning,
    )
    logger.info(f"Explicit timestep conditioning: {use_timestep_conditioning}")
    
    # Freeze models
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, unet=unet, controlnet=controlnet, scheduler=scheduler,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision inference we cast model weights to half-precision.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def main(args, enable_xformers_memory_efficient_attention=True,):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")
            ori_width, ori_height = validation_image.size
            validation_image, padding = pad_image_to_multiple(validation_image, multiple=8)
            pad_width, pad_height = padding
            width, height = validation_image.size
            validation_depth = None
            needs_coc_depth = args.diffusion_process in ("coc_blur", "coc_endpoint") or (
                args.diffusion_process == "coc_image_latent"
                and args.coc_image_latent_reverse == "recompute_prev"
            )
            if needs_coc_depth and args.use_depth:
                depth_name = resolve_depth_path(args.depth_path, image_name)
                if depth_name is None:
                    raise FileNotFoundError(f"Cannot find a depth map for `{image_name}`.")
                validation_depth = Image.open(depth_name).convert("L")
                validation_depth, depth_padding = pad_image_to_multiple(validation_depth, multiple=8)
                if validation_depth.size != validation_image.size:
                    raise ValueError(f"Depth size mismatch between `{image_name}` and `{depth_name}`.")
                if depth_padding != padding:
                    raise ValueError(f"Depth padding mismatch between `{image_name}` and `{depth_name}`.")

            if pad_width or pad_height:
                print(f'input size: {ori_height}x{ori_width}, padded to {height}x{width}')
            else:
                print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    image = pipeline(
                            validation_image, depth=validation_depth, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            conditioning_scale=args.conditioning_scale,
                            start_point=args.start_point,
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            diffusion_process=args.diffusion_process,
                            coc_focus_depth=args.coc_focus_depth,
                            coc_focus_width=args.coc_focus_width,
                            coc_focus_depth_min=args.coc_focus_depth_min,
                            coc_focus_depth_max=args.coc_focus_depth_max,
                            coc_focus_width_min=args.coc_focus_width_min,
                            coc_focus_width_max=args.coc_focus_width_max,
                            coc_global_blur_min=args.coc_global_blur_min,
                            coc_global_blur_max=args.coc_global_blur_max,
                            coc_max_radius=args.coc_max_radius,
                            coc_gamma=args.coc_gamma,
                            coc_schedule_power=args.coc_schedule_power,
                            coc_global_blur_at_max=args.coc_global_blur_at_max,
                            coc_depth_blur_strength=args.coc_depth_blur_strength,
                            coc_inference_start=args.coc_inference_start,
                            coc_image_latent_reverse=args.coc_image_latent_reverse,
                            coc_image_latent_timestep_spacing=args.coc_image_latent_timestep_spacing,
                            coc_image_latent_normalize_start=args.coc_image_latent_normalize_start,
                            coc_noise_normalization=args.coc_noise_normalization,
                            coc_noise_normalization_eps=args.coc_noise_normalization_eps,
                            start_blur_sigma=args.start_blur_sigma,
                            start_blur_kernel_size=args.start_blur_kernel_size,
                            update_blend=args.update_blend,
                            start_steps=args.start_steps,
                            args=args,
                        ).images[0]
                
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if pad_width or pad_height:
                    image = image.crop((0, 0, ori_width, ori_height))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--depth_path", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--vae_decoder_tiled_size",
        type=int,
        default=0,
        help="VAE decoder tile size. Set <=0 to disable VAE decoder tiling and avoid seam artifacts.",
    )
    parser.add_argument(
        "--vae_encoder_tiled_size",
        type=int,
        default=0,
        help="VAE encoder tile size. Set <=0 to disable VAE encoder tiling and avoid seam artifacts.",
    )
    parser.add_argument(
        "--latent_tiled_size",
        type=int,
        default=256,
        help="Latent tile area threshold. Set <=0 to disable latent tiling and avoid tile seam artifacts.",
    )
    parser.add_argument("--latent_tiled_overlap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument(
        "--diffusion_process",
        type=str,
        choices=["gaussian", "coc_blur", "paired_endpoint", "coc_endpoint", "coc_image_latent"],
        default="gaussian",
    )
    parser.add_argument("--coc_focus_depth", type=float, default=0.7)
    parser.add_argument("--coc_focus_width", type=float, default=0.0)
    parser.add_argument("--coc_focus_depth_min", type=float, default=0.1)
    parser.add_argument("--coc_focus_depth_max", type=float, default=0.9)
    parser.add_argument("--coc_focus_width_min", type=float, default=0.0)
    parser.add_argument("--coc_focus_width_max", type=float, default=0.12)
    parser.add_argument("--coc_global_blur_min", type=float, default=0.0)
    parser.add_argument("--coc_global_blur_max", type=float, default=1.0)
    parser.add_argument("--coc_max_radius", type=float, default=2.5)
    parser.add_argument("--coc_gamma", type=float, default=1.5)
    parser.add_argument(
        "--coc_schedule_power",
        type=float,
        default=3.0,
        help="Power for timestep-to-blur mapping. Larger values keep early timesteps cleaner and accelerate blur near the end.",
    )
    parser.add_argument("--coc_global_blur_at_max", type=float, default=0.0)
    parser.add_argument("--coc_depth_blur_strength", type=float, default=1.0)
    parser.add_argument(
        "--coc_image_latent_reverse",
        type=str,
        choices=["scheduler", "recompute_prev", "source_endpoint"],
        default="scheduler",
        help=(
            "Reverse process for coc_image_latent. scheduler uses the official scheduler.step. "
            "recompute_prev estimates clean latent first, then rebuilds the previous CoC blur latent; "
            "it requires --use_depth to take effect. source_endpoint anchors the reverse trajectory to "
            "the observed source latent and does not require depth."
        ),
    )
    parser.add_argument(
        "--coc_image_latent_timestep_spacing",
        type=str,
        choices=["scheduler", "full_range"],
        default="scheduler",
        help=(
            "Timestep schedule for coc_image_latent inference. scheduler keeps diffusers defaults; "
            "full_range uses [T-1 ... 0], so one-step inference really starts from the strongest degradation."
        ),
    )
    parser.add_argument(
        "--coc_image_latent_normalize_start",
        action="store_true",
        help="Enable sample-wise normalization for the initial source latent in coc_image_latent inference.",
    )
    parser.add_argument(
        "--coc_noise_normalization",
        type=str,
        choices=["none", "sample"],
        default="sample",
        help="Normalization mode used when recomputing CoC blur latents during inference.",
    )
    parser.add_argument("--coc_noise_normalization_eps", type=float, default=1e-6)
    parser.add_argument(
        "--coc_inference_start",
        type=str,
        choices=["latent_max_blur", "encoded_input", "gaussian_blur"],
        default="encoded_input",
        help=(
            "CoC inference start. encoded_input uses the source/blurred image latent directly. "
            "latent_max_blur additionally applies full-image maximum CoC blur in latent space."
        ),
    )
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--start_blur_sigma", type=float, default=8.0)
    parser.add_argument("--start_blur_kernel_size", type=int, default=None)
    parser.add_argument("--update_blend", type=float, default=1.0)
    parser.add_argument(
        "--timestep_conditioning",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Use explicit timestep embeddings. auto keeps Gaussian and CoC image-latent on, and cold-diffusion CoC paths off.",
    )
    args = parser.parse_args()
    main(args)
