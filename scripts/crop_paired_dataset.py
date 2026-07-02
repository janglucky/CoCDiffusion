#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_hw(values, name):
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError(f"`{name}` expects one value or two values: height width.")


def list_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"`{image_dir}` is not a directory.")

    image_index = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stem in image_index:
            raise ValueError(f"Duplicate image stem `{path.stem}` found in `{image_dir}`.")
        image_index[path.stem] = path
    return image_index


def sorted_items(image_index):
    return sorted(image_index.items(), key=lambda item: item[1].name)


def get_positions(length, patch, stride, cover_edges):
    if length < patch:
        return []
    positions = list(range(0, length - patch + 1, stride))
    if not positions:
        positions = [0]
    if cover_edges and positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def pad_to_min_size(image, min_width, min_height):
    width, height = image.size
    pad_width = max(min_width - width, 0)
    pad_height = max(min_height - height, 0)
    if pad_width == 0 and pad_height == 0:
        return image

    array = np.asarray(image)
    if array.ndim == 2:
        pad_widths = ((0, pad_height), (0, pad_width))
    else:
        pad_widths = ((0, pad_height), (0, pad_width), (0, 0))
    padded = np.pad(array, pad_widths, mode="edge")
    return Image.fromarray(padded)


def save_patch(image, box, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(output_path)


def resolve_dirs(args):
    source_dir = Path(args.input_root) / args.source_name
    target_dir = Path(args.input_root) / args.target_name
    depth_dir = Path(args.input_root) / args.depth_name

    if args.include_depth == "yes" and not depth_dir.is_dir():
        raise FileNotFoundError(f"`--include_depth yes` requires `{depth_dir}`.")
    use_depth = args.include_depth == "yes" or (args.include_depth == "auto" and depth_dir.is_dir())
    return source_dir, target_dir, depth_dir if use_depth else None


def build_pairs(source_dir, target_dir, depth_dir, pairing_mode, depth_pairing_mode):
    source_index = list_images(source_dir)
    target_index = list_images(target_dir)

    pairs = []
    source_keys = set(source_index)
    target_keys = set(target_index)
    same_name_pairs = source_keys == target_keys

    if pairing_mode == "name" or (pairing_mode == "auto" and same_name_pairs):
        if not same_name_pairs:
            missing_sources = sorted(target_keys - source_keys)
            missing_targets = sorted(source_keys - target_keys)
            raise ValueError(
                f"Mismatched source/target pairs: "
                f"{len(missing_sources)} missing source, {len(missing_targets)} missing target."
            )
        for key in sorted(source_keys):
            pairs.append([key, source_index[key], target_index[key], None])
        print(f"[pairing] source/target matched by filename: {len(pairs)} pairs", flush=True)
    elif pairing_mode in ("auto", "sorted"):
        if len(source_index) != len(target_index):
            raise ValueError(
                f"Cannot pair source/target by sorted order because counts differ: "
                f"{len(source_index)} source vs {len(target_index)} target."
            )
        for (source_key, source_path), (_, target_path) in zip(sorted_items(source_index), sorted_items(target_index)):
            pairs.append([source_key, source_path, target_path, None])
        print(
            "[pairing] source/target matched by sorted order "
            f"because filenames do not match exactly: {len(pairs)} pairs",
            flush=True,
        )
    else:
        raise ValueError(f"Unknown pairing mode `{pairing_mode}`.")

    if not pairs:
        raise ValueError(f"No paired images found between `{source_dir}` and `{target_dir}`.")

    if depth_dir is not None:
        depth_index = list_images(depth_dir)
        missing_depth = [pair[0] for pair in pairs if pair[0] not in depth_index]

        if depth_pairing_mode == "source_name" or (depth_pairing_mode == "auto" and not missing_depth):
            if missing_depth:
                raise ValueError(f"Missing {len(missing_depth)} source-name depth maps in `{depth_dir}`.")
            for pair in pairs:
                pair[3] = depth_index[pair[0]]
            print(f"[pairing] depth matched by source filename: {len(pairs)} maps", flush=True)
        elif depth_pairing_mode in ("auto", "sorted"):
            if len(depth_index) != len(pairs):
                raise ValueError(
                    f"Cannot pair depth by sorted order because counts differ: "
                    f"{len(depth_index)} depth vs {len(pairs)} pairs."
                )
            for pair, (_, depth_path) in zip(pairs, sorted_items(depth_index)):
                pair[3] = depth_path
            print(f"[pairing] depth matched by sorted order: {len(pairs)} maps", flush=True)
        else:
            raise ValueError(f"Unknown depth pairing mode `{depth_pairing_mode}`.")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Crop source/target or source/target/depth paired datasets into aligned patches."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Dataset root containing source/target[/depth].")
    parser.add_argument("--output_root", type=str, required=True, help="Output dataset root.")
    parser.add_argument("--source_name", type=str, default="source")
    parser.add_argument("--target_name", type=str, default="target")
    parser.add_argument("--depth_name", type=str, default="depth")
    parser.add_argument("--patch_size", type=int, nargs="+", default=[512], help="Patch size: one value or H W.")
    parser.add_argument("--stride", type=int, nargs="+", default=None, help="Stride: one value or H W.")
    parser.add_argument(
        "--include_depth",
        type=str,
        choices=["auto", "yes", "no"],
        default="auto",
        help="Crop depth maps if present, require them, or ignore them.",
    )
    parser.add_argument(
        "--pairing_mode",
        type=str,
        choices=["auto", "name", "sorted"],
        default="auto",
        help="Pair source/target by filename, sorted order, or filename first then sorted fallback.",
    )
    parser.add_argument(
        "--depth_pairing_mode",
        type=str,
        choices=["auto", "source_name", "sorted"],
        default="auto",
        help="Pair depth by source filename first, or by sorted order.",
    )
    parser.add_argument(
        "--edge_policy",
        type=str,
        choices=["cover", "drop"],
        default="cover",
        help="`cover` adds edge windows; `drop` only keeps regular sliding windows.",
    )
    parser.add_argument(
        "--pad_if_needed",
        action="store_true",
        help="Replicate-pad images smaller than patch size before cropping.",
    )
    parser.add_argument("--output_ext", type=str, default=".png", help="Output extension, for example .png.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of source images to process.")
    args = parser.parse_args()

    patch_h, patch_w = parse_hw(args.patch_size, "--patch_size")
    if args.stride is None:
        stride_h, stride_w = patch_h, patch_w
    else:
        stride_h, stride_w = parse_hw(args.stride, "--stride")

    if patch_h <= 0 or patch_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("Patch size and stride must be positive.")
    if not args.output_ext.startswith("."):
        args.output_ext = f".{args.output_ext}"

    source_dir, target_dir, depth_dir = resolve_dirs(args)
    pairs = build_pairs(source_dir, target_dir, depth_dir, args.pairing_mode, args.depth_pairing_mode)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output_root = Path(args.output_root)
    source_out = output_root / args.source_name
    target_out = output_root / args.target_name
    depth_out = output_root / args.depth_name if depth_dir is not None else None

    total_patches = 0
    skipped = 0
    cover_edges = args.edge_policy == "cover"

    for image_idx, (key, source_path, target_path, depth_path) in enumerate(pairs):
        source = Image.open(source_path)
        target = Image.open(target_path)
        depth = Image.open(depth_path) if depth_path is not None else None

        if source.size != target.size:
            raise ValueError(f"Size mismatch for `{key}`: source {source.size}, target {target.size}.")
        if depth is not None and source.size != depth.size:
            raise ValueError(f"Size mismatch for `{key}`: source {source.size}, depth {depth.size}.")

        if args.pad_if_needed:
            source = pad_to_min_size(source, patch_w, patch_h)
            target = pad_to_min_size(target, patch_w, patch_h)
            if depth is not None:
                depth = pad_to_min_size(depth, patch_w, patch_h)

        width, height = source.size
        xs = get_positions(width, patch_w, stride_w, cover_edges)
        ys = get_positions(height, patch_h, stride_h, cover_edges)
        if not xs or not ys:
            skipped += 1
            print(
                f"[skip] {key}: image size {height}x{width} is smaller than patch {patch_h}x{patch_w}.",
                flush=True,
            )
            continue

        for y in ys:
            for x in xs:
                box = (x, y, x + patch_w, y + patch_h)
                patch_name = f"{key}_y{y:05d}_x{x:05d}{args.output_ext}"
                save_patch(source, box, source_out / patch_name)
                save_patch(target, box, target_out / patch_name)
                if depth is not None:
                    save_patch(depth, box, depth_out / patch_name)
                total_patches += 1

        print(f"[{image_idx + 1}/{len(pairs)}] {key}: {len(xs) * len(ys)} patches", flush=True)

    print(f"Done. Wrote {total_patches} patches to `{output_root}`. Skipped {skipped} images.", flush=True)


if __name__ == "__main__":
    main()
