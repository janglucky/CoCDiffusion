import torch

from schedulers.coc_blur_scheduler import CoCBlurScheduler


class CoCImageLatentScheduler:
    """CoC image-space degradation with latent-space diffusion states.

    Forward process:
        z_0 = E(x_0)
        z_blur,t = E(CoC_t(x_0, depth))
        x_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * z_blur,t

    CoC blur is applied in image space, while the diffusion state x_t remains
    in latent space.
    """

    def __init__(self, coc_blur_scheduler: CoCBlurScheduler):
        self.coc_blur_scheduler = coc_blur_scheduler
        self.num_train_timesteps = coc_blur_scheduler.num_train_timesteps

    def _default_depth(self, image):
        return torch.zeros(
            (image.shape[0], 1, image.shape[-2], image.shape[-1]),
            device=image.device,
            dtype=image.dtype,
        )

    def _prepare_depth(self, depth, image):
        if depth is None:
            return self._default_depth(image)
        return self.coc_blur_scheduler._ensure_depth(depth, image)

    def _prepare_params(self, image, focus_depth=None, focus_width=None, global_blur_floor=None, no_depth=False):
        batch_size = image.shape[0]
        focus_depth = self.coc_blur_scheduler._ensure_sample_param(focus_depth, image, batch_size, "focus_depth")
        focus_width = self.coc_blur_scheduler._ensure_sample_param(focus_width, image, batch_size, "focus_width")
        global_blur_floor = self.coc_blur_scheduler._ensure_sample_param(
            global_blur_floor,
            image,
            batch_size,
            "global_blur_floor",
        )

        if no_depth:
            if focus_depth is None:
                focus_depth = torch.full((batch_size,), 0.5, device=image.device, dtype=image.dtype)
            if focus_width is None:
                focus_width = torch.zeros((batch_size,), device=image.device, dtype=image.dtype)
            if global_blur_floor is None:
                global_blur_floor = torch.ones((batch_size,), device=image.device, dtype=image.dtype)
            return focus_depth, focus_width, global_blur_floor

        if focus_depth is None or focus_width is None or global_blur_floor is None:
            sampled_focus_depth, sampled_focus_width, sampled_global_blur_floor = (
                self.coc_blur_scheduler.sample_dof_params(image)
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

    def _alpha(self, timesteps, sample):
        blur_scale = self.coc_blur_scheduler._timestep_to_scale(timesteps, sample)
        return (1.0 - blur_scale).clamp(0.0, 1.0)

    def mix_latents(self, clean_latents, blurred_latents, timesteps):
        alpha = self._alpha(timesteps, clean_latents)
        return alpha.sqrt() * clean_latents + (1.0 - alpha).sqrt() * blurred_latents

    def blur_image(
        self,
        image,
        depth,
        timesteps,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        no_depth = depth is None
        depth = self._prepare_depth(depth, image)
        focus_depth, focus_width, global_blur_floor = self._prepare_params(
            image,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
            no_depth=no_depth,
        )
        return self.coc_blur_scheduler.add_blur(
            image,
            depth,
            timesteps,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        ).clamp(-1.0, 1.0)

    def add_degradation(
        self,
        clean_image,
        clean_latents,
        depth,
        timesteps,
        encode_image_to_latents,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        blurred_image = self.blur_image(
            clean_image,
            depth,
            timesteps,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        blurred_latents = encode_image_to_latents(blurred_image)
        return self.mix_latents(clean_latents, blurred_latents, timesteps)

    def reconstruct_degradation(
        self,
        predicted_clean,
        depth,
        timesteps,
        decode_latents_to_image,
        encode_image_to_latents,
        focus_depth=None,
        focus_width=None,
        global_blur_floor=None,
    ):
        predicted_image = decode_latents_to_image(predicted_clean).clamp(-1.0, 1.0)
        blurred_image = self.blur_image(
            predicted_image,
            depth,
            timesteps,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        blurred_latents = encode_image_to_latents(blurred_image)
        return self.mix_latents(predicted_clean, blurred_latents, timesteps)

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
        decode_latents_to_image,
        encode_image_to_latents,
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

        degraded_t = self.reconstruct_degradation(
            predicted_clean,
            depth,
            t,
            decode_latents_to_image,
            encode_image_to_latents,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        degraded_prev = self.reconstruct_degradation(
            predicted_clean,
            depth,
            t_prev,
            decode_latents_to_image,
            encode_image_to_latents,
            focus_depth=focus_depth,
            focus_width=focus_width,
            global_blur_floor=global_blur_floor,
        )
        return sample - degraded_t + degraded_prev
