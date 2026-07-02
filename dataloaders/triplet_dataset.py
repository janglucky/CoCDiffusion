import os
from pathlib import Path

from PIL import Image
from torch.utils import data as data
from torchvision import transforms


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(image_dir):
    return sorted(
        str(path)
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _build_image_index(image_dir):
    image_index = {}
    for image_path in _list_images(image_dir):
        image_key = Path(image_path).stem
        if image_key in image_index:
            raise ValueError(f"Duplicate image key `{image_key}` found in `{image_dir}`.")
        image_index[image_key] = image_path
    return image_index


def _resolve_triplet_dirs(root_folder):
    input_dir = os.path.join(root_folder, "source")
    target_dir = os.path.join(root_folder, "target")
    depth_dir = os.path.join(root_folder, "depth")

    missing_dirs = [path for path in (input_dir, target_dir, depth_dir) if not os.path.isdir(path)]
    if missing_dirs:
        raise FileNotFoundError(
            f"`{root_folder}` must contain source/target/depth directories. Missing: {missing_dirs}"
        )
    return input_dir, target_dir, depth_dir


class TripletImageDataset(data.Dataset):
    def __init__(
        self,
        root_folders=None,
    ):
        super(TripletImageDataset, self).__init__()

        self.triplets = []

        root_folders = [folder.strip() for folder in root_folders.split(",") if folder.strip()]
        if not root_folders:
            raise ValueError("`root_folders` must provide at least one dataset root.")

        for root_folder in root_folders:
            input_dir, target_dir, depth_dir = _resolve_triplet_dirs(root_folder)
            input_index = _build_image_index(input_dir)
            target_index = _build_image_index(target_dir)
            depth_index = _build_image_index(depth_dir)

            common_keys = sorted(set(input_index.keys()) & set(target_index.keys()) & set(depth_index.keys()))
            if not common_keys:
                raise ValueError(f"No triplet images found in `{root_folder}`.")

            missing_inputs = sorted((set(target_index.keys()) | set(depth_index.keys())) - set(input_index.keys()))
            missing_targets = sorted((set(input_index.keys()) | set(depth_index.keys())) - set(target_index.keys()))
            missing_depths = sorted((set(input_index.keys()) | set(target_index.keys())) - set(depth_index.keys()))
            if missing_inputs or missing_targets or missing_depths:
                raise ValueError(
                    f"Mismatched triplets in `{root_folder}`: "
                    f"{len(missing_inputs)} missing inputs, "
                    f"{len(missing_targets)} missing targets, "
                    f"{len(missing_depths)} missing depths."
                )

            self.triplets.extend(
                (input_index[key], target_index[key], depth_index[key]) for key in common_keys
            )

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.depth_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        lq_path, gt_path, depth_path = self.triplets[index]

        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = self.img_preproc(gt_img)
        lq_img = Image.open(lq_path).convert("RGB")
        lq_img = self.img_preproc(lq_img)
        depth_img = Image.open(depth_path).convert("L")
        depth_img = self.depth_preproc(depth_img)

        if gt_img.shape != lq_img.shape:
            raise ValueError(f"Image size mismatch between `{lq_path}` and `{gt_path}`.")
        if depth_img.shape[-2:] != lq_img.shape[-2:]:
            raise ValueError(f"Depth size mismatch between `{lq_path}` and `{depth_path}`.")

        example = dict()
        example["conditioning_pixel_values"] = lq_img
        example["pixel_values"] = gt_img * 2.0 - 1.0
        example["depth_pixel_values"] = depth_img

        return example

    def __len__(self):
        return len(self.triplets)
