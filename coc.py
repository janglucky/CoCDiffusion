import numpy as np
import torch
import torch.nn.functional as F


class DefocusRenderer:
    """Depth-aware defocus renderer used as a deterministic blur degradation."""

    def __init__(
        self,
        focus_depth=0.7,
        max_radius=20.0,
        gamma=1.5,
        radii=(0, 1, 2, 3, 5, 7, 10, 14, 20),
    ):
        self.focus_depth = focus_depth
        self.max_radius = float(max_radius)
        self.gamma = gamma
        self.radii = [float(radius) for radius in radii]
        if self.radii[0] != 0:
            raise ValueError("`radii` must start with 0 so the renderer can represent the clean image.")
        if sorted(self.radii) != self.radii:
            raise ValueError("`radii` must be sorted in ascending order.")

    def depth_to_coc(self, depth, max_radius=None):
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        coc = torch.abs(depth - self.focus_depth)
        coc = coc.pow(self.gamma)
        coc = coc * float(self.max_radius if max_radius is None else max_radius)

        return coc

    def disk_kernel(self, radius, device=None, dtype=None):
        if radius < 1:
            return torch.ones(1, 1, device=device, dtype=dtype or torch.float32)

        r = int(np.ceil(radius))
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing="ij",
        )

        mask = (x**2 + y**2) <= radius**2
        kernel = mask.to(dtype=dtype or torch.float32)
        kernel /= kernel.sum().clamp_min(1e-6)

        return kernel

    def disk_blur(self, image, radius):
        if radius < 1:
            return image

        kernel = self.disk_kernel(radius, device=image.device, dtype=image.dtype)
        k = kernel.shape[0]
        kernel = kernel.view(1, 1, k, k).repeat(image.shape[1], 1, 1, 1)

        # Replicate padding avoids dark borders when large blur radii are used.
        padded = F.pad(image, (k // 2, k // 2, k // 2, k // 2), mode="replicate")
        return F.conv2d(padded, kernel, groups=image.shape[1])

    def build_blur_stack(self, image, radii=None):
        radii = self.radii if radii is None else radii
        blur_levels = [self.disk_blur(image, radius) for radius in radii]
        return torch.stack(blur_levels, dim=1)

    def interpolate(self, blur_stack, coc, radii=None):
        radii = self.radii if radii is None else radii
        _, n, _, _, _ = blur_stack.shape
        if n != len(radii):
            raise ValueError("`blur_stack` and `radii` must have the same number of levels.")

        output = blur_stack[:, 0]
        for i in range(n - 1):
            r0 = radii[i]
            r1 = radii[i + 1]
            mask = (coc >= r0) & (coc < r1)
            alpha = ((coc - r0) / max(r1 - r0, 1e-6)).clamp(0, 1)
            img = (1 - alpha) * blur_stack[:, i] + alpha * blur_stack[:, i + 1]
            output = torch.where(mask.expand_as(output), img, output)

        output = torch.where((coc >= radii[-1]).expand_as(output), blur_stack[:, -1], output)
        return output

    def scaled_radii(self, radius_scale):
        return [radius * float(radius_scale) for radius in self.radii]

    def render(self, image, depth, radius_scale=1.0, max_radius=None):
        effective_max_radius = self.max_radius if max_radius is None else float(max_radius)
        effective_max_radius *= float(radius_scale)
        radii = self.scaled_radii(radius_scale)

        coc = self.depth_to_coc(depth, max_radius=effective_max_radius)
        blur_stack = self.build_blur_stack(image, radii=radii)
        return self.interpolate(blur_stack, coc, radii=radii)
