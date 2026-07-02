#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from coc import DefocusRenderer
from schedulers.coc_blur_scheduler import CoCBlurScheduler


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_depth_path(depth_path, image_path, source_path=None):
    depth_path = Path(depth_path)
    image_path = Path(image_path)
    source_path = Path(source_path) if source_path else None

    if depth_path.is_file():
        return depth_path
    if not depth_path.is_dir():
        raise FileNotFoundError(f"`{depth_path}` is not a file or directory.")

    candidates = []
    if source_path is not None:
        candidates.append(depth_path / source_path.name)
        candidates.extend(sorted(depth_path.glob(f"{source_path.stem}.*")))
    candidates.append(depth_path / image_path.name)
    candidates.extend(sorted(depth_path.glob(f"{image_path.stem}.*")))

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve depth for `{image_path}` from `{depth_path}`. "
        "Pass a depth image file directly or use --source_path when image/depth names differ."
    )


def resize_for_vis(image, max_size, resample):
    if max_size is None or max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, resample)


def pad_to_multiple(image, multiple=8):
    width, height = image.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple
    if pad_width == 0 and pad_height == 0:
        return image

    array = np.asarray(image)
    if array.ndim == 2:
        pad_widths = ((0, pad_height), (0, pad_width))
    else:
        pad_widths = ((0, pad_height), (0, pad_width), (0, 0))
    return Image.fromarray(np.pad(array, pad_widths, mode="edge"))


def image_to_tensor(image, device, dtype=torch.float32):
    array = np.asarray(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def tensor_to_image(tensor):
    tensor = tensor.detach().float().clamp(0, 1)[0]
    array = tensor.permute(1, 2, 0).cpu().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    array = (array * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def save_grid(images, labels, output_path, cell_width=None):
    if not images:
        return

    label_h = 28
    pad = 8
    cell_width = cell_width or max(image.width for image in images)
    cell_height = max(image.height for image in images)

    cells = []
    for image, label in zip(images, labels):
        canvas = Image.new("RGB", (cell_width, cell_height + label_h), "white")
        image_rgb = image.convert("RGB")
        x = (cell_width - image_rgb.width) // 2
        y = label_h + (cell_height - image_rgb.height) // 2
        canvas.paste(image_rgb, (x, y))
        ImageDraw.Draw(canvas).text((6, 7), label, fill=(0, 0, 0))
        cells.append(canvas)

    grid = Image.new("RGB", (len(cells) * cell_width + (len(cells) - 1) * pad, cell_height + label_h), "white")
    x = 0
    for cell in cells:
        grid.paste(cell, (x, 0))
        x += cell_width + pad
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def save_matrix_grid(rows, row_labels, col_labels, output_path, corner_label="focus / blur"):
    if not rows or not rows[0]:
        return

    pad = 6
    label_w = 126
    label_h = 34
    cell_width = max(image.width for row in rows for image in row)
    cell_height = max(image.height for row in rows for image in row)
    width = label_w + len(col_labels) * cell_width + (len(col_labels) + 1) * pad
    height = label_h + len(rows) * cell_height + (len(rows) + 1) * pad
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((pad, pad + 8), corner_label, fill=(0, 0, 0))

    for col_idx, label in enumerate(col_labels):
        x = label_w + pad + col_idx * (cell_width + pad)
        draw.text((x + 6, pad + 8), label, fill=(0, 0, 0))

    for row_idx, (row, row_label) in enumerate(zip(rows, row_labels)):
        y = label_h + pad + row_idx * (cell_height + pad)
        draw.text((pad, y + cell_height // 2 - 7), row_label, fill=(0, 0, 0))
        for col_idx, image in enumerate(row):
            x = label_w + pad + col_idx * (cell_width + pad)
            image_rgb = image.convert("RGB")
            grid.paste(image_rgb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def evenly_spaced(start, end, count):
    if count <= 1:
        return [float((start + end) * 0.5)]
    return [float(value) for value in np.linspace(float(start), float(end), int(count))]


def matrix_focus_conditions(args):
    depth_values = evenly_spaced(args.coc_focus_depth_min, args.coc_focus_depth_max, args.matrix_rows)
    width_values = evenly_spaced(args.coc_focus_width_min, args.coc_focus_width_max, args.matrix_rows)
    global_blur_values = evenly_spaced(args.coc_global_blur_min, args.coc_global_blur_max, args.matrix_rows)
    depth_mid = float((args.coc_focus_depth_min + args.coc_focus_depth_max) * 0.5)
    width_mid = float((args.coc_focus_width_min + args.coc_focus_width_max) * 0.5)
    global_blur_mid = float((args.coc_global_blur_min + args.coc_global_blur_max) * 0.5)

    if args.matrix_axis == "focus_width":
        return [(depth_mid, width, global_blur_mid) for width in width_values]
    if args.matrix_axis == "global_blur":
        return [(depth_mid, width_mid, global_blur) for global_blur in global_blur_values]
    if args.matrix_axis == "both":
        return list(zip(depth_values, width_values, global_blur_values))
    return [(depth, width_mid, global_blur_mid) for depth in depth_values]


def save_radius_maps(
    depth,
    renderer,
    output_dir,
    timesteps,
    scales,
    max_radius,
    focus_depth,
    focus_width,
    global_blur_floor,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_img = tensor_to_image(depth)
    depth_img.save(output_dir / "depth_normalized.png")

    radius_images = []
    labels = []
    with torch.no_grad():
        defocus = renderer.depth_defocus(depth, focus_depth=focus_depth, focus_width=focus_width)
        tensor_to_image(defocus).save(output_dir / "defocus_normalized.png")
        for timestep, scale in zip(timesteps, scales):
            radius = renderer.depth_to_coc(
                depth,
                max_radius=max_radius,
                radius_scale=scale,
                focus_depth=focus_depth,
                focus_width=focus_width,
                global_blur_floor=global_blur_floor,
            )
            radius_vis = radius / max(float(max_radius), 1e-6)
            radius_image = tensor_to_image(radius_vis)
            radius_image.save(output_dir / f"radius_map_t_{int(timestep):04d}.png")
            radius_images.append(radius_image)
            labels.append(f"t={int(timestep)} s={scale:.2f}")

    save_grid(radius_images, labels, output_dir / "grid_radius_map.png")


def radius_stats(depth, renderer, max_radius, radius_scale, focus_depth, focus_width, global_blur_floor):
    with torch.no_grad():
        radius = renderer.depth_to_coc(
            depth,
            max_radius=max_radius,
            radius_scale=radius_scale,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
    return radius.amin().item(), radius.mean().item(), radius.amax().item()


def visualize_image_matrix(args, image_tensor, depth_tensor, scheduler, output_dir, image_max_radius):
    if args.matrix_rows <= 0:
        return

    timestep_tensors = [torch.tensor([timestep], device=image_tensor.device, dtype=torch.long) for timestep in args.timesteps]
    scales = [scheduler._timestep_to_scale(t, image_tensor).flatten()[0].item() for t in timestep_tensors]
    focus_conditions = matrix_focus_conditions(args)
    col_labels = [f"t={int(t)} s={scale:.2f}" for t, scale in zip(args.timesteps, scales)]

    image_rows = []
    radius_rows = []
    row_labels = []
    for focus_depth_value, focus_width_value, global_blur_value in focus_conditions:
        focus_depth = torch.tensor([focus_depth_value], device=image_tensor.device, dtype=image_tensor.dtype)
        focus_width = torch.tensor([focus_width_value], device=image_tensor.device, dtype=image_tensor.dtype)
        global_blur_floor = torch.tensor([global_blur_value], device=image_tensor.device, dtype=image_tensor.dtype)
        image_row = []
        radius_row = []
        for timestep_tensor, scale in zip(timestep_tensors, scales):
            blurred = scheduler.add_blur(
                image_tensor,
                depth_tensor,
                timestep_tensor,
                focus_depth=focus_depth,
                focus_width=focus_width,
                global_blur_floor=global_blur_floor,
            )
            radius = scheduler.renderer.depth_to_coc(
                depth_tensor,
                max_radius=image_max_radius,
                radius_scale=scale,
                focus_depth=focus_depth_value,
                focus_width=focus_width_value,
                global_blur_floor=global_blur_value,
            )
            image_row.append(tensor_to_image(blurred))
            radius_row.append(tensor_to_image(radius / max(float(image_max_radius), 1e-6)))

        image_rows.append(image_row)
        radius_rows.append(radius_row)
        row_labels.append(f"f={focus_depth_value:.2f} w={focus_width_value:.2f} g={global_blur_value:.2f}")

    save_matrix_grid(image_rows, row_labels, col_labels, output_dir / "matrix_image_space.png")
    save_matrix_grid(radius_rows, row_labels, col_labels, output_dir / "matrix_radius_map.png")


def visualize_image_space(args, image, depth, device, output_dir):
    image_tensor = image_to_tensor(image, device=device, dtype=torch.float32)
    depth_tensor = image_to_tensor(depth, device=device, dtype=torch.float32)
    image_max_radius = args.image_max_radius
    if image_max_radius is None:
        image_max_radius = args.coc_max_radius * args.image_radius_multiplier

    scheduler = CoCBlurScheduler(
        num_train_timesteps=args.num_train_timesteps,
        focus_depth=args.coc_focus_depth,
        focus_width=args.coc_focus_width,
        max_radius=image_max_radius,
        gamma=args.coc_gamma,
        schedule_power=args.coc_schedule_power,
        global_blur_at_max=args.coc_global_blur_at_max,
        depth_blur_strength=args.coc_depth_blur_strength,
        focus_depth_min=args.coc_focus_depth_min,
        focus_depth_max=args.coc_focus_depth_max,
        focus_width_min=args.coc_focus_width_min,
        focus_width_max=args.coc_focus_width_max,
    )

    mode_dir = output_dir / "image_space"
    mode_dir.mkdir(parents=True, exist_ok=True)
    focus_depth, focus_width, global_blur_floor = scheduler.sample_dof_params(image_tensor)
    focus_depth_value = float(focus_depth[0].item())
    focus_width_value = float(focus_width[0].item())
    global_blur_value = float(global_blur_floor[0].item())
    timestep_tensors = [torch.tensor([timestep], device=device, dtype=torch.long) for timestep in args.timesteps]
    scales = [scheduler._timestep_to_scale(t, image_tensor).flatten()[0].item() for t in timestep_tensors]
    save_radius_maps(
        depth_tensor,
        scheduler.renderer,
        mode_dir,
        args.timesteps,
        scales,
        image_max_radius,
        focus_depth=focus_depth_value,
        focus_width=focus_width_value,
        global_blur_floor=global_blur_value,
    )

    images = [image]
    labels = [f"clean f={focus_depth_value:.2f} w={focus_width_value:.2f} g={global_blur_value:.2f}"]
    for timestep, timestep_tensor, scale in zip(args.timesteps, timestep_tensors, scales):
        blurred = scheduler.add_blur(
            image_tensor,
            depth_tensor,
            timestep_tensor,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        r_min, r_mean, r_max = radius_stats(
            depth_tensor,
            scheduler.renderer,
            image_max_radius,
            scale,
            focus_depth=focus_depth_value,
            focus_width=focus_width_value,
            global_blur_floor=global_blur_value,
        )
        blurred_image = tensor_to_image(blurred)
        blurred_image.save(mode_dir / f"t_{int(timestep):04d}_scale_{scale:.4f}.png")
        images.append(blurred_image)
        labels.append(f"t={int(timestep)} s={scale:.2f} r={r_min:.1f}/{r_mean:.1f}/{r_max:.1f}")

    save_grid(images, labels, mode_dir / "grid_image_space.png")
    visualize_image_matrix(args, image_tensor, depth_tensor, scheduler, mode_dir, image_max_radius)


def visualize_latent_matrix(args, latents, latent_depth, scheduler, vae, dtype, output_dir):
    if args.matrix_rows <= 0:
        return

    timestep_tensors = [torch.tensor([timestep], device=latents.device, dtype=torch.long) for timestep in args.timesteps]
    scales = [scheduler._timestep_to_scale(t, latents.float()).flatten()[0].item() for t in timestep_tensors]
    focus_conditions = matrix_focus_conditions(args)
    col_labels = [f"t={int(t)} s={scale:.2f}" for t, scale in zip(args.timesteps, scales)]

    image_rows = []
    radius_rows = []
    row_labels = []
    with torch.no_grad():
        for focus_depth_value, focus_width_value, global_blur_value in focus_conditions:
            focus_depth = torch.tensor([focus_depth_value], device=latents.device, dtype=latents.dtype)
            focus_width = torch.tensor([focus_width_value], device=latents.device, dtype=latents.dtype)
            global_blur_floor = torch.tensor([global_blur_value], device=latents.device, dtype=latents.dtype)
            image_row = []
            radius_row = []
            for timestep_tensor, scale in zip(timestep_tensors, scales):
                blurred_latents = scheduler.add_blur(
                    latents.float(),
                    latent_depth,
                    timestep_tensor,
                    focus_depth=focus_depth,
                    focus_width=focus_width,
                    global_blur_floor=global_blur_floor,
                )
                decoded = vae.decode(blurred_latents.to(dtype=dtype) / vae.config.scaling_factor, return_dict=False)[0]
                decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
                radius = scheduler.renderer.depth_to_coc(
                    latent_depth,
                    max_radius=args.coc_max_radius,
                    radius_scale=scale,
                    focus_depth=focus_depth_value,
                    focus_width=focus_width_value,
                    global_blur_floor=global_blur_value,
                )
                image_row.append(tensor_to_image(decoded))
                radius_row.append(tensor_to_image(radius / max(float(args.coc_max_radius), 1e-6)))

            image_rows.append(image_row)
            radius_rows.append(radius_row)
            row_labels.append(f"f={focus_depth_value:.2f} w={focus_width_value:.2f} g={global_blur_value:.2f}")

    save_matrix_grid(image_rows, row_labels, col_labels, output_dir / "matrix_latent_space.png")
    save_matrix_grid(radius_rows, row_labels, col_labels, output_dir / "matrix_radius_map.png")


def visualize_latent_space(args, image, depth, device, output_dir):
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" and args.fp16_vae else torch.float32
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    vae.to(device=device, dtype=dtype)
    vae.eval()

    image = pad_to_multiple(image, multiple=8)
    depth = pad_to_multiple(depth, multiple=8)
    image_tensor = image_to_tensor(image, device=device, dtype=dtype)
    depth_tensor = image_to_tensor(depth, device=device, dtype=dtype)

    scheduler = CoCBlurScheduler(
        num_train_timesteps=args.num_train_timesteps,
        focus_depth=args.coc_focus_depth,
        focus_width=args.coc_focus_width,
        max_radius=args.coc_max_radius,
        gamma=args.coc_gamma,
        schedule_power=args.coc_schedule_power,
        global_blur_at_max=args.coc_global_blur_at_max,
        depth_blur_strength=args.coc_depth_blur_strength,
        focus_depth_min=args.coc_focus_depth_min,
        focus_depth_max=args.coc_focus_depth_max,
        focus_width_min=args.coc_focus_width_min,
        focus_width_max=args.coc_focus_width_max,
    )

    mode_dir = output_dir / "latent_space"
    mode_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        latents = vae.encode(image_tensor * 2.0 - 1.0).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        latent_depth = F.interpolate(depth_tensor.float(), size=latents.shape[-2:], mode="bilinear", align_corners=False)
        focus_depth, focus_width, global_blur_floor = scheduler.sample_dof_params(latents)
        focus_depth_value = float(focus_depth[0].item())
        focus_width_value = float(focus_width[0].item())
        global_blur_value = float(global_blur_floor[0].item())
        timestep_tensors = [torch.tensor([timestep], device=device, dtype=torch.long) for timestep in args.timesteps]
        scales = [scheduler._timestep_to_scale(t, latents.float()).flatten()[0].item() for t in timestep_tensors]
        save_radius_maps(
            latent_depth,
            scheduler.renderer,
            mode_dir,
            args.timesteps,
            scales,
            args.coc_max_radius,
            focus_depth=focus_depth_value,
            focus_width=focus_width_value,
            global_blur_floor=global_blur_value,
        )

        clean_decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        clean_decoded = (clean_decoded / 2.0 + 0.5).clamp(0, 1)

        images = [tensor_to_image(clean_decoded)]
        labels = [f"vae clean f={focus_depth_value:.2f} w={focus_width_value:.2f} g={global_blur_value:.2f}"]
        for timestep, timestep_tensor, scale in zip(args.timesteps, timestep_tensors, scales):
            blurred_latents = scheduler.add_blur(
                latents.float(),
                latent_depth,
                timestep_tensor,
                focus_depth=focus_depth,
                focus_width=focus_width,
                global_blur_floor=global_blur_floor,
            )
            decoded = vae.decode(blurred_latents.to(dtype=dtype) / vae.config.scaling_factor, return_dict=False)[0]
            decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
            r_min, r_mean, r_max = radius_stats(
                latent_depth,
                scheduler.renderer,
                args.coc_max_radius,
                scale,
                focus_depth=focus_depth_value,
                focus_width=focus_width_value,
                global_blur_floor=global_blur_value,
            )
            decoded_image = tensor_to_image(decoded)
            decoded_image.save(mode_dir / f"t_{int(timestep):04d}_scale_{scale:.4f}.png")
            images.append(decoded_image)
            labels.append(f"t={int(timestep)} s={scale:.2f} r={r_min:.2f}/{r_mean:.2f}/{r_max:.2f}")

    save_grid(images, labels, mode_dir / "grid_latent_space.png")
    visualize_latent_matrix(args, latents, latent_depth, scheduler, vae, dtype, mode_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the CoC forward blur process used by CoCDiffusion."
    )
    parser.add_argument("--image_path", type=str, required=True, help="Clean image to degrade. For training parity, use target/clean images.")
    parser.add_argument("--depth_path", type=str, required=True, help="Depth image or a directory containing depth maps.")
    parser.add_argument("--source_path", type=str, default=None, help="Optional source image name used to resolve depth when target/depth names differ.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="/home/gd09385/models/stable-diffusion-2-base")
    parser.add_argument("--mode", type=str, choices=["image", "latent", "both"], default="both")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 100, 250, 500, 750, 999])
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--coc_focus_depth", type=float, default=0.7)
    parser.add_argument("--coc_focus_width", type=float, default=0.0)
    parser.add_argument("--coc_focus_depth_min", type=float, default=0.1)
    parser.add_argument("--coc_focus_depth_max", type=float, default=0.9)
    parser.add_argument("--coc_focus_width_min", type=float, default=0.0)
    parser.add_argument("--coc_focus_width_max", type=float, default=0.12)
    parser.add_argument("--coc_global_blur_min", type=float, default=0.0)
    parser.add_argument("--coc_global_blur_max", type=float, default=1.0)
    parser.add_argument("--coc_max_radius", type=float, default=2.5, help="Latent-space max CoC radius used during training.")
    parser.add_argument("--coc_gamma", type=float, default=1.5)
    parser.add_argument(
        "--coc_schedule_power",
        type=float,
        default=3.0,
        help="Power for timestep-to-blur mapping. Larger values keep early timesteps cleaner and accelerate blur near the end.",
    )
    parser.add_argument("--coc_global_blur_at_max", type=float, default=0.0)
    parser.add_argument("--coc_depth_blur_strength", type=float, default=1.0)
    parser.add_argument("--image_max_radius", type=float, default=None, help="Image-space CoC radius. Defaults to coc_max_radius * image_radius_multiplier.")
    parser.add_argument("--image_radius_multiplier", type=float, default=8.0)
    parser.add_argument("--max_size", type=int, default=768, help="Resize longest side for visualization. Use 0 to keep original size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the random focus/depth-of-field visualization condition.")
    parser.add_argument("--matrix_rows", type=int, default=5, help="Rows in the 2D visualization matrix. Use 0 to disable.")
    parser.add_argument(
        "--matrix_axis",
        type=str,
        choices=["focus_depth", "focus_width", "global_blur", "both"],
        default="global_blur",
        help="Vertical axis for matrix visualization.",
    )
    parser.add_argument("--fp16_vae", action="store_true", help="Use fp16 VAE in latent mode on CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    depth_path = resolve_depth_path(args.depth_path, image_path, args.source_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")
    image = resize_for_vis(image, args.max_size, Image.BICUBIC)
    depth = resize_for_vis(depth, args.max_size, Image.BILINEAR)
    if depth.size != image.size:
        depth = depth.resize(image.size, Image.BILINEAR)

    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    metadata = {
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "timesteps": args.timesteps,
        "num_train_timesteps": args.num_train_timesteps,
        "coc_focus_depth": args.coc_focus_depth,
        "coc_focus_width": args.coc_focus_width,
        "coc_focus_depth_min": args.coc_focus_depth_min,
        "coc_focus_depth_max": args.coc_focus_depth_max,
        "coc_focus_width_min": args.coc_focus_width_min,
        "coc_focus_width_max": args.coc_focus_width_max,
        "coc_global_blur_min": args.coc_global_blur_min,
        "coc_global_blur_max": args.coc_global_blur_max,
        "coc_max_radius_latent": args.coc_max_radius,
        "image_max_radius": args.image_max_radius
        if args.image_max_radius is not None
        else args.coc_max_radius * args.image_radius_multiplier,
        "coc_gamma": args.coc_gamma,
        "coc_schedule_power": args.coc_schedule_power,
        "coc_global_blur_at_max": args.coc_global_blur_at_max,
        "coc_depth_blur_strength": args.coc_depth_blur_strength,
        "seed": args.seed,
        "matrix_rows": args.matrix_rows,
        "matrix_axis": args.matrix_axis,
        "resized_size": image.size,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if args.mode in ("image", "both"):
        visualize_image_space(args, image, depth, device, output_dir)
    if args.mode in ("latent", "both"):
        visualize_latent_space(args, image, depth, device, output_dir)

    print(f"Saved CoC forward visualization to: {output_dir}")
    if args.mode in ("image", "both"):
        print(f"Image-space grid: {output_dir / 'image_space' / 'grid_image_space.png'}")
        if args.matrix_rows > 0:
            print(f"Image-space matrix: {output_dir / 'image_space' / 'matrix_image_space.png'}")
    if args.mode in ("latent", "both"):
        print(f"Latent-space grid: {output_dir / 'latent_space' / 'grid_latent_space.png'}")
        if args.matrix_rows > 0:
            print(f"Latent-space matrix: {output_dir / 'latent_space' / 'matrix_latent_space.png'}")


if __name__ == "__main__":
    main()
