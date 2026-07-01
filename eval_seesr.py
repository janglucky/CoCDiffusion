import argparse
import json
from pathlib import Path

import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_image_index(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"`{path}` does not exist.")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"`{path}` is not a supported image file.")
        return {path.stem: path}

    image_index = {}
    for image_path in sorted(path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if image_path.stem in image_index:
            raise ValueError(f"Duplicate image key `{image_path.stem}` found in `{path}`.")
        image_index[image_path.stem] = image_path

    if not image_index:
        raise ValueError(f"No images found in `{path}`.")

    return image_index


def load_rgb_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SeeSR deblurring outputs with PSNR/SSIM.")
    parser.add_argument("--prediction_path", type=str, required=True, help="Prediction image or directory.")
    parser.add_argument("--target_path", type=str, required=True, help="Target image or directory.")
    parser.add_argument("--crop_border", type=int, default=0, help="Crop border before metric calculation.")
    parser.add_argument(
        "--test_y_channel",
        action="store_true",
        help="Evaluate on the Y channel instead of RGB.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the metric summary as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prediction_index = build_image_index(args.prediction_path)
    target_index = build_image_index(args.target_path)

    common_keys = sorted(set(prediction_index.keys()) & set(target_index.keys()))
    missing_predictions = sorted(set(target_index.keys()) - set(prediction_index.keys()))
    missing_targets = sorted(set(prediction_index.keys()) - set(target_index.keys()))

    if not common_keys:
        raise ValueError("No matching image pairs found between prediction and target paths.")

    psnr_scores = []
    ssim_scores = []
    per_image_results = []

    for image_key in common_keys:
        prediction_image = load_rgb_image(prediction_index[image_key])
        target_image = load_rgb_image(target_index[image_key])

        if prediction_image.shape != target_image.shape:
            raise ValueError(
                f"Image size mismatch for `{image_key}`: "
                f"{prediction_image.shape} vs {target_image.shape}."
            )

        psnr_value = float(
            calculate_psnr(
                prediction_image,
                target_image,
                crop_border=args.crop_border,
                test_y_channel=args.test_y_channel,
            )
        )
        ssim_value = float(
            calculate_ssim(
                prediction_image,
                target_image,
                crop_border=args.crop_border,
                test_y_channel=args.test_y_channel,
            )
        )

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)
        per_image_results.append(
            {
                "name": image_key,
                "prediction": str(prediction_index[image_key]),
                "target": str(target_index[image_key]),
                "psnr": psnr_value,
                "ssim": ssim_value,
            }
        )

        if args.verbose:
            print(f"{image_key}: PSNR={psnr_value:.4f} dB, SSIM={ssim_value:.6f}")

    summary = {
        "num_pairs": len(common_keys),
        "mean_psnr": float(np.mean(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "crop_border": args.crop_border,
        "test_y_channel": args.test_y_channel,
        "missing_predictions": missing_predictions,
        "missing_targets": missing_targets,
        "per_image": per_image_results,
    }

    print(f"Matched pairs: {summary['num_pairs']}")
    print(f"Mean PSNR: {summary['mean_psnr']:.4f} dB")
    print(f"Mean SSIM: {summary['mean_ssim']:.6f}")
    if missing_predictions:
        print(f"Missing predictions: {len(missing_predictions)}")
    if missing_targets:
        print(f"Missing targets: {len(missing_targets)}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
