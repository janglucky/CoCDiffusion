import numpy as np
import torch
import torch.nn.functional as F


class DefocusRenderer:
    """Depth-aware defocus renderer used as a deterministic blur degradation."""

    def __init__(
        self,
        focus_depth=0.7,
        focus_width=0.0,
        max_radius=20.0,
        gamma=1.5,
        radii=(0, 1, 2, 3, 5, 7, 10, 14, 20),
        global_blur_at_max=0.0,
        depth_blur_strength=1.0,
    ):
        self.focus_depth = float(focus_depth)
        self.focus_width = float(focus_width)
        self.max_radius = float(max_radius)
        self.gamma = gamma
        self.global_blur_at_max = float(global_blur_at_max)
        self.depth_blur_strength = float(depth_blur_strength)
        self.radii = [float(radius) for radius in radii]
        if self.radii[0] != 0:
            raise ValueError("`radii` must start with 0 so the renderer can represent the clean image.")
        if sorted(self.radii) != self.radii:
            raise ValueError("`radii` must be sorted in ascending order.")

    def normalize_depth(self, depth):
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        return (depth - depth_min) / (depth_max - depth_min + 1e-6)

    def depth_defocus(self, depth, focus_depth=None, focus_width=None):
        depth = self.normalize_depth(depth)
        focus_depth = self.focus_depth if focus_depth is None else float(focus_depth)
        focus_depth = min(max(focus_depth, 0.0), 1.0)
        focus_width = self.focus_width if focus_width is None else float(focus_width)
        focus_width = min(max(focus_width, 0.0), 1.0)

        distance = (torch.abs(depth - focus_depth) - focus_width).clamp_min(0)
        max_distance = max(focus_depth - focus_width, 1.0 - focus_depth - focus_width, 1e-6)
        defocus = (distance / max_distance).clamp(0, 1)
        return defocus.pow(self.gamma)

    def depth_to_coc(
        self,
        depth,
        max_radius=None,
        radius_scale=1.0,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        """Return per-pixel blur radius.

        `radius_scale` is the diffusion-time blur strength and maps directly to
        the CoC radius. The focal depth/DOF shape is an independent degradation
        condition and should not be tied to the timestep.
        """
        max_radius = float(self.max_radius if max_radius is None else max_radius)
        radius_scale = float(radius_scale)
        radius_scale = min(max(radius_scale, 0.0), 1.0)

        global_blur_floor = self.global_blur_at_max if global_blur_floor is None else float(global_blur_floor)
        global_blur_floor = min(max(global_blur_floor, 0.0), 1.0)

        raw_defocus = self.depth_defocus(depth, focus_depth=focus_depth, focus_width=focus_width)
        defocus = (raw_defocus * self.depth_blur_strength).clamp(0, 1)
        radius_factor = radius_scale * defocus
        if global_blur_floor > 0:
            global_floor = global_blur_floor * radius_scale
            radius_factor = torch.maximum(radius_factor, torch.full_like(radius_factor, global_floor))
        return radius_factor.clamp(0, 1) * max_radius

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
        if radius <= 1e-6:
            return image
        if radius < 1:
            fully_blurred = self.disk_blur(image, 1.0)
            return image * (1.0 - float(radius)) + fully_blurred * float(radius)

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

    def radii_for_max_radius(self, max_radius):
        max_radius = float(max_radius)
        if self.max_radius <= 1e-6:
            return self.radii
        scale = max_radius / self.max_radius
        return [radius * scale for radius in self.radii]

    def render(
        self,
        image,
        depth,
        radius_scale=1.0,
        max_radius=None,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        effective_max_radius = self.max_radius if max_radius is None else float(max_radius)
        radii = self.radii_for_max_radius(effective_max_radius)

        coc = self.depth_to_coc(
            depth,
            max_radius=effective_max_radius,
            radius_scale=radius_scale,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        blur_stack = self.build_blur_stack(image, radii=radii)
        return self.interpolate(blur_stack, coc, radii=radii)
