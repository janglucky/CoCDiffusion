import os
from pathlib import Path
from PIL import Image

from torchvision import transforms
from torch.utils import data as data


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


def _resolve_pair_dirs(root_folder):
    for input_dir_name, target_dir_name in (("source", "target"), ("sr_bicubic", "gt")):
        input_dir = os.path.join(root_folder, input_dir_name)
        target_dir = os.path.join(root_folder, target_dir_name)
        if os.path.isdir(input_dir) and os.path.isdir(target_dir):
            return input_dir, target_dir

    raise FileNotFoundError(
        f"`{root_folder}` must contain either `source/target` or `sr_bicubic/gt` directories."
    )


class PairedImageDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
    ):
        super(PairedImageDataset, self).__init__()

        self.pairs = []

        root_folders = [folder.strip() for folder in root_folders.split(',') if folder.strip()]
        if not root_folders:
            raise ValueError("`root_folders` must provide at least one dataset root.")

        for root_folder in root_folders:
            input_dir, target_dir = _resolve_pair_dirs(root_folder)
            input_index = _build_image_index(input_dir)
            target_index = _build_image_index(target_dir)
            common_keys = sorted(set(input_index.keys()) & set(target_index.keys()))

            if not common_keys:
                raise ValueError(f"No paired images found between `{input_dir}` and `{target_dir}`.")

            missing_inputs = sorted(set(target_index.keys()) - set(input_index.keys()))
            missing_targets = sorted(set(input_index.keys()) - set(target_index.keys()))
            if missing_inputs or missing_targets:
                raise ValueError(
                    f"Mismatched pairs in `{root_folder}`: "
                    f"{len(missing_inputs)} missing inputs, {len(missing_targets)} missing targets."
                )

            self.pairs.extend((input_index[key], target_index[key]) for key in common_keys)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        lq_path, gt_path = self.pairs[index]

        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if gt_img.shape != lq_img.shape:
            raise ValueError(f"Image size mismatch between `{lq_path}` and `{gt_path}`.")

        example = dict()
        example["conditioning_pixel_values"] = lq_img
        example["pixel_values"] = gt_img * 2.0 - 1.0

        return example

    def __len__(self):
        return len(self.pairs)
