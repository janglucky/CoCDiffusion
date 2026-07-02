import torch


class PairedEndpointScheduler:
    """Cold-diffusion scheduler between clean and paired degraded endpoints.

    Forward process:
        z_t = (1 - alpha_t) * clean + alpha_t * degraded

    At t=0 this is the clean latent, and at the final timestep this is the
    paired source/degraded latent used as the inference starting point.
    """

    def __init__(self, num_train_timesteps=1000, schedule_power=1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule_power = float(schedule_power)

    def _timestep_to_alpha(self, timesteps, sample):
        timesteps = torch.as_tensor(timesteps, device=sample.device)
        timesteps = timesteps.to(dtype=sample.dtype)
        denom = max(self.num_train_timesteps - 1, 1)
        alpha = (timesteps / denom).clamp(0, 1).pow(self.schedule_power)
        return alpha.view(-1, 1, 1, 1)

    def add_degradation(self, clean, degraded_endpoint, timesteps):
        alpha = self._timestep_to_alpha(timesteps, clean)
        degraded_endpoint = degraded_endpoint.to(device=clean.device, dtype=clean.dtype)
        return (1.0 - alpha) * clean + alpha * degraded_endpoint

    def previous_timestep(self, timestep, inference_timesteps):
        index = (inference_timesteps == timestep).nonzero(as_tuple=False)
        if len(index) == 0:
            raise ValueError(f"Timestep {timestep} was not found in inference timesteps.")
        index = int(index[0].item())
        if index + 1 >= len(inference_timesteps):
            return torch.zeros_like(timestep)
        return inference_timesteps[index + 1]

    def step(self, predicted_clean, sample, degraded_endpoint, timestep, inference_timesteps):
        batch_size = predicted_clean.shape[0]
        timestep_value = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        t = torch.full(
            (batch_size,),
            timestep_value,
            device=predicted_clean.device,
            dtype=torch.long,
        )
        prev_timestep = self.previous_timestep(timestep, inference_timesteps)
        t_prev = torch.full_like(t, int(prev_timestep.item()))

        degraded_t = self.add_degradation(predicted_clean, degraded_endpoint, t)
        degraded_prev = self.add_degradation(predicted_clean, degraded_endpoint, t_prev)
        return sample - degraded_t + degraded_prev
