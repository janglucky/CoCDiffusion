import torch
import torch.nn.functional as F

from coc import DefocusRenderer


class CoCBlurScheduler:
    """Cold-diffusion scheduler whose forward process progressively applies CoC blur."""

    def __init__(
        self,
        num_train_timesteps=1000,
        focus_depth=0.7,
        focus_width=0.0,
        max_radius=2.5,
        gamma=1.5,
        radii=None,
        schedule_power=3.0,
        global_blur_at_max=0.0,
        depth_blur_strength=1.0,
        focus_depth_min=None,
        focus_depth_max=None,
        focus_width_min=None,
        focus_width_max=None,
        global_blur_min=None,
        global_blur_max=None,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule_power = float(schedule_power)
        self.focus_depth = float(focus_depth)
        self.focus_width = float(focus_width)
        self.focus_depth_min = focus_depth_min
        self.focus_depth_max = focus_depth_max
        self.focus_width_min = focus_width_min
        self.focus_width_max = focus_width_max
        self.global_blur_at_max = float(global_blur_at_max)
        self.global_blur_min = global_blur_min
        self.global_blur_max = global_blur_max
        if radii is None:
            base_fractions = (0, 1 / 20, 2 / 20, 3 / 20, 5 / 20, 7 / 20, 10 / 20, 14 / 20, 1)
            radii = tuple(float(max_radius) * fraction for fraction in base_fractions)
        self.renderer = DefocusRenderer(
            focus_depth=focus_depth,
            focus_width=focus_width,
            max_radius=max_radius,
            gamma=gamma,
            radii=radii,
            global_blur_at_max=global_blur_at_max,
            depth_blur_strength=depth_blur_strength,
        )

    def _timestep_to_scale(self, timesteps, sample):
        timesteps = torch.as_tensor(timesteps, device=sample.device)
        timesteps = timesteps.to(dtype=sample.dtype)
        denom = max(self.num_train_timesteps - 1, 1)
        scale = (timesteps / denom).clamp(0, 1).pow(self.schedule_power)
        return scale.view(-1, 1, 1, 1)

    def _sample_scalar(self, default, lower, upper, batch_size, sample):
        if lower is None and upper is None:
            return torch.full((batch_size,), float(default), device=sample.device, dtype=sample.dtype)

        lower = float(default if lower is None else lower)
        upper = float(lower if upper is None else upper)
        if upper < lower:
            raise ValueError(f"Invalid random range: min={lower} is greater than max={upper}.")
        if abs(upper - lower) <= 1e-12:
            return torch.full((batch_size,), lower, device=sample.device, dtype=sample.dtype)
        return torch.empty((batch_size,), device=sample.device, dtype=sample.dtype).uniform_(lower, upper)

    def sample_dof_params(self, sample, batch_size=None):
        batch_size = sample.shape[0] if batch_size is None else int(batch_size)
        focus_depth = self._sample_scalar(
            self.focus_depth,
            self.focus_depth_min,
            self.focus_depth_max,
            batch_size,
            sample,
        )
        focus_width = self._sample_scalar(
            self.focus_width,
            self.focus_width_min,
            self.focus_width_max,
            batch_size,
            sample,
        )
        global_blur_floor = self._sample_scalar(
            self.global_blur_at_max,
            self.global_blur_min,
            self.global_blur_max,
            batch_size,
            sample,
        )
        return focus_depth.clamp(0, 1), focus_width.clamp_min(0), global_blur_floor.clamp(0, 1)

    def sample_focus_params(self, sample, batch_size=None):
        focus_depth, focus_width, _ = self.sample_dof_params(sample, batch_size=batch_size)
        return focus_depth, focus_width

    def _ensure_sample_param(self, value, sample, batch_size, name):
        if value is None:
            return None
        value = torch.as_tensor(value, device=sample.device, dtype=sample.dtype).flatten()
        if value.numel() == 1:
            value = value.repeat(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"`{name}` must have 1 value or {batch_size} values, got {value.numel()}.")
        return value

    def _ensure_depth(self, depth, sample):
        if depth is None:
            raise ValueError("`depth` is required for CoC blur diffusion.")

        depth = depth.to(device=sample.device, dtype=sample.dtype)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] != 1:
            depth = depth[:, :1]
        if depth.shape[-2:] != sample.shape[-2:]:
            depth = F.interpolate(depth, size=sample.shape[-2:], mode="bilinear", align_corners=False)
        if depth.shape[0] == 1 and sample.shape[0] > 1:
            depth = depth.repeat(sample.shape[0], 1, 1, 1)
        return depth

    def add_blur(self, clean, depth, timesteps, focus_depth=None, focus_width=None, global_blur_floor=None):
        depth = self._ensure_depth(depth, clean)
        scales = self._timestep_to_scale(timesteps, clean)
        focus_depth = self._ensure_sample_param(focus_depth, clean, clean.shape[0], "focus_depth")
        focus_width = self._ensure_sample_param(focus_width, clean, clean.shape[0], "focus_width")
        global_blur_floor = self._ensure_sample_param(
            global_blur_floor,
            clean,
            clean.shape[0],
            "global_blur_floor",
        )
        if focus_depth is None or focus_width is None or global_blur_floor is None:
            sampled_focus_depth, sampled_focus_width, sampled_global_blur_floor = self.sample_dof_params(clean)
            if focus_depth is None:
                focus_depth = sampled_focus_depth
            if focus_width is None:
                focus_width = sampled_focus_width
            if global_blur_floor is None:
                global_blur_floor = sampled_global_blur_floor
        output = []
        for idx in range(clean.shape[0]):
            output.append(
                self.renderer.render(
                    clean[idx : idx + 1],
                    depth[idx : idx + 1],
                    radius_scale=float(scales[idx].item()),
                    focus_depth=float(focus_depth[idx].item()),
                    focus_width=float(focus_width[idx].item()),
                    global_blur_floor=float(global_blur_floor[idx].item()),
                )
            )
        return torch.cat(output, dim=0)

    def previous_timestep(self, timestep, inference_timesteps):
        index = (inference_timesteps == timestep).nonzero(as_tuple=False)
        if len(index) == 0:
            raise ValueError(f"Timestep {timestep} was not found in inference timesteps.")
        index = int(index[0].item())
        if index + 1 >= len(inference_timesteps):
            return torch.zeros_like(timestep)
        return inference_timesteps[index + 1]

    def step(
        self,
        predicted_clean,
        sample,
        depth,
        timestep,
        inference_timesteps,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        depth = self._ensure_depth(depth, predicted_clean)
        t = torch.full(
            (predicted_clean.shape[0],),
            int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep),
            device=predicted_clean.device,
            dtype=torch.long,
        )
        prev_timestep = self.previous_timestep(timestep, inference_timesteps)
        t_prev = torch.full_like(t, int(prev_timestep.item()))
        focus_depth = self._ensure_sample_param(focus_depth, predicted_clean, predicted_clean.shape[0], "focus_depth")
        focus_width = self._ensure_sample_param(focus_width, predicted_clean, predicted_clean.shape[0], "focus_width")
        global_blur_floor = self._ensure_sample_param(
            global_blur_floor,
            predicted_clean,
            predicted_clean.shape[0],
            "global_blur_floor",
        )
        if focus_depth is None or focus_width is None or global_blur_floor is None:
            sampled_focus_depth, sampled_focus_width, sampled_global_blur_floor = self.sample_dof_params(predicted_clean)
            if focus_depth is None:
                focus_depth = sampled_focus_depth
            if focus_width is None:
                focus_width = sampled_focus_width
            if global_blur_floor is None:
                global_blur_floor = sampled_global_blur_floor

        degraded_t = self.add_blur(
            predicted_clean,
            depth,
            t,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        degraded_prev = self.add_blur(
            predicted_clean,
            depth,
            t_prev,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        return sample - degraded_t + degraded_prev
