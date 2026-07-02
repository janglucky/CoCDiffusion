import torch
import torch.nn.functional as F

from coc import DefocusRenderer


class CoCBlurScheduler:
    """Cold-diffusion scheduler whose forward process progressively applies CoC blur."""

    def __init__(
        self,
        num_train_timesteps=1000,
        focus_depth=0.7,
        max_radius=2.5,
        gamma=1.5,
        radii=None,
        schedule_power=1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule_power = float(schedule_power)
        if radii is None:
            base_fractions = (0, 1 / 20, 2 / 20, 3 / 20, 5 / 20, 7 / 20, 10 / 20, 14 / 20, 1)
            radii = tuple(float(max_radius) * fraction for fraction in base_fractions)
        self.renderer = DefocusRenderer(
            focus_depth=focus_depth,
            max_radius=max_radius,
            gamma=gamma,
            radii=radii,
        )

    def _timestep_to_scale(self, timesteps, sample):
        timesteps = torch.as_tensor(timesteps, device=sample.device)
        timesteps = timesteps.to(dtype=sample.dtype)
        denom = max(self.num_train_timesteps - 1, 1)
        scale = (timesteps / denom).clamp(0, 1).pow(self.schedule_power)
        return scale.view(-1, 1, 1, 1)

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

    def add_blur(self, clean, depth, timesteps):
        depth = self._ensure_depth(depth, clean)
        scales = self._timestep_to_scale(timesteps, clean)
        output = []
        for idx in range(clean.shape[0]):
            output.append(
                self.renderer.render(
                    clean[idx : idx + 1],
                    depth[idx : idx + 1],
                    radius_scale=float(scales[idx].item()),
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

    def step(self, predicted_clean, sample, depth, timestep, inference_timesteps):
        depth = self._ensure_depth(depth, predicted_clean)
        t = torch.full(
            (predicted_clean.shape[0],),
            int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep),
            device=predicted_clean.device,
            dtype=torch.long,
        )
        prev_timestep = self.previous_timestep(timestep, inference_timesteps)
        t_prev = torch.full_like(t, int(prev_timestep.item()))

        degraded_t = self.add_blur(predicted_clean, depth, t)
        degraded_prev = self.add_blur(predicted_clean, depth, t_prev)
        return sample - degraded_t + degraded_prev
