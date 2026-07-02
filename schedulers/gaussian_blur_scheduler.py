import math

import torch
import torch.nn.functional as F


class GaussianBlurScheduler:
    """Cold-diffusion scheduler whose degradation is progressive Gaussian blur."""

    def __init__(
        self,
        num_train_timesteps=1000,
        max_sigma=2.5,
        kernel_size=None,
        schedule_power=1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.max_sigma = float(max_sigma)
        self.kernel_size = kernel_size
        self.schedule_power = float(schedule_power)

    def _timestep_to_sigma(self, timesteps, sample):
        timesteps = torch.as_tensor(timesteps, device=sample.device)
        timesteps = timesteps.to(dtype=sample.dtype)
        denom = max(self.num_train_timesteps - 1, 1)
        sigma = (timesteps / denom).clamp(0, 1).pow(self.schedule_power) * self.max_sigma
        return sigma.view(-1)

    def _kernel_size(self, sigma):
        if self.kernel_size is not None:
            kernel_size = int(self.kernel_size)
        else:
            kernel_size = int(math.ceil(float(sigma) * 6))
            kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _gaussian_kernel(self, sigma, device, dtype):
        if sigma <= 1e-6:
            return torch.ones(1, 1, 1, 1, device=device, dtype=dtype)

        kernel_size = self._kernel_size(sigma)
        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def _blur_one(self, sample, sigma):
        if sigma <= 1e-6:
            return sample

        kernel = self._gaussian_kernel(sigma, sample.device, sample.dtype)
        kernel_size = kernel.shape[-1]
        kernel = kernel.repeat(sample.shape[1], 1, 1, 1)
        padded = F.pad(
            sample,
            (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
            mode="replicate",
        )
        return F.conv2d(padded, kernel, groups=sample.shape[1])

    def add_blur(self, clean, timesteps):
        sigmas = self._timestep_to_sigma(timesteps, clean)
        output = []
        for idx in range(clean.shape[0]):
            output.append(self._blur_one(clean[idx : idx + 1], float(sigmas[idx].item())))
        return torch.cat(output, dim=0)

    def previous_timestep(self, timestep, inference_timesteps):
        index = (inference_timesteps == timestep).nonzero(as_tuple=False)
        if len(index) == 0:
            raise ValueError(f"Timestep {timestep} was not found in inference timesteps.")
        index = int(index[0].item())
        if index + 1 >= len(inference_timesteps):
            return torch.zeros_like(timestep)
        return inference_timesteps[index + 1]

    def step(self, predicted_clean, sample, timestep, inference_timesteps):
        t = torch.full(
            (predicted_clean.shape[0],),
            int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep),
            device=predicted_clean.device,
            dtype=torch.long,
        )
        prev_timestep = self.previous_timestep(timestep, inference_timesteps)
        t_prev = torch.full_like(t, int(prev_timestep.item()))

        degraded_t = self.add_blur(predicted_clean, t)
        degraded_prev = self.add_blur(predicted_clean, t_prev)
        return sample - degraded_t + degraded_prev
