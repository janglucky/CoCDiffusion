import torch

from schedulers.coc_blur_scheduler import CoCBlurScheduler


class CoCEndpointScheduler:
    """CoC-guided cold diffusion with the paired real blur as the endpoint.

    Forward process:
        c_t = CoC_t(clean, depth)
        c_T = CoC_T(clean, depth)
        z_t = c_t + alpha_t * (degraded_endpoint - c_T)

    This keeps the CoC blur trajectory while forcing the final endpoint to be
    the actual source/blurred latent used at inference.
    """

    def __init__(self, coc_blur_scheduler: CoCBlurScheduler):
        self.coc_blur_scheduler = coc_blur_scheduler
        self.num_train_timesteps = coc_blur_scheduler.num_train_timesteps

    def _default_depth(self, sample):
        return torch.zeros(
            (sample.shape[0], 1, sample.shape[-2], sample.shape[-1]),
            device=sample.device,
            dtype=sample.dtype,
        )

    def _prepare_depth(self, depth, sample):
        if depth is None:
            return self._default_depth(sample)
        return self.coc_blur_scheduler._ensure_depth(depth, sample)

    def _prepare_params(self, sample, focus_depth=None, focus_width=None, global_blur_floor=None, no_depth=False):
        batch_size = sample.shape[0]
        focus_depth = self.coc_blur_scheduler._ensure_sample_param(focus_depth, sample, batch_size, "focus_depth")
        focus_width = self.coc_blur_scheduler._ensure_sample_param(focus_width, sample, batch_size, "focus_width")
        global_blur_floor = self.coc_blur_scheduler._ensure_sample_param(
            global_blur_floor,
            sample,
            batch_size,
            "global_blur_floor",
        )

        if no_depth:
            if focus_depth is None:
                focus_depth = torch.full((batch_size,), 0.5, device=sample.device, dtype=sample.dtype)
            if focus_width is None:
                focus_width = torch.zeros((batch_size,), device=sample.device, dtype=sample.dtype)
            if global_blur_floor is None:
                global_blur_floor = torch.ones((batch_size,), device=sample.device, dtype=sample.dtype)
            return focus_depth, focus_width, global_blur_floor

        if focus_depth is None or focus_width is None or global_blur_floor is None:
            sampled_focus_depth, sampled_focus_width, sampled_global_blur_floor = (
                self.coc_blur_scheduler.sample_dof_params(sample)
            )
            if focus_depth is None:
                focus_depth = sampled_focus_depth
            if focus_width is None:
                focus_width = sampled_focus_width
            if global_blur_floor is None:
                global_blur_floor = sampled_global_blur_floor
        return focus_depth, focus_width, global_blur_floor

    def sample_dof_params(self, sample, batch_size=None):
        return self.coc_blur_scheduler.sample_dof_params(sample, batch_size=batch_size)

    def add_degradation(
        self,
        clean,
        degraded_endpoint,
        depth,
        timesteps,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        no_depth = depth is None
        depth = self._prepare_depth(depth, clean)
        focus_depth, focus_width, global_blur_floor = self._prepare_params(
            clean,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
            no_depth=no_depth,
        )

        degraded_endpoint = degraded_endpoint.to(device=clean.device, dtype=clean.dtype)
        alpha = self.coc_blur_scheduler._timestep_to_scale(timesteps, clean)
        final_timesteps = torch.full(
            (clean.shape[0],),
            self.num_train_timesteps - 1,
            device=clean.device,
            dtype=torch.long,
        )

        coc_t = self.coc_blur_scheduler.add_blur(
            clean,
            depth,
            timesteps,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        coc_final = self.coc_blur_scheduler.add_blur(
            clean,
            depth,
            final_timesteps,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        return coc_t + alpha * (degraded_endpoint - coc_final)

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
        degraded_endpoint,
        depth,
        timestep,
        inference_timesteps,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
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

        degraded_t = self.add_degradation(
            predicted_clean,
            degraded_endpoint,
            depth,
            t,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        degraded_prev = self.add_degradation(
            predicted_clean,
            degraded_endpoint,
            depth,
            t_prev,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        return sample - degraded_t + degraded_prev
