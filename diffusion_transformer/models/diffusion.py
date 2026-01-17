"""
Diffusion process utilities including noise schedules and sampling.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DiffusionProcess:
    """
    Handles the forward and reverse diffusion process.
    Supports multiple noise schedules and sampling strategies.
    """

    def __init__(
            self,
            num_timesteps=1000,
            schedule_type="cosine",
            beta_start=0.0001,
            beta_end=0.02,
            device="cuda"
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            schedule_type: Type of noise schedule ('linear', 'cosine', 'quadratic')
            beta_start: Starting beta value (for linear schedule)
            beta_end: Ending beta value (for linear schedule)
            device: Device to use
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = device

        # Create noise schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        betas = betas.to(device)

        # Pre-compute values for diffusion
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to clean data.

        Args:
            x_0: Clean data (batch, seq_len, dim)
            t: Timesteps (batch,)
            noise: Optional pre-generated noise

        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.

        Args:
            x_t: Noisy data at timestep t
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted x_0
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
                self.posterior_mean_coef1[t].unsqueeze(-1).unsqueeze(-1) * x_0 +
                self.posterior_mean_coef2[t].unsqueeze(-1).unsqueeze(-1) * x_t
        )
        posterior_variance = self.posterior_variance[t].unsqueeze(-1).unsqueeze(-1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].unsqueeze(-1).unsqueeze(-1)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, model, x_t, t, mask=None, clip_denoised=True):
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t)

        Args:
            model: Denoising model
            x_t: Noisy data at timestep t
            t: Timesteps
            mask: Optional mask
            clip_denoised: Whether to clip predicted x_0

        Returns:
            model_mean, model_variance, model_log_variance
        """
        # Predict noise
        noise_pred = model(x_t, t, mask)

        # Predict x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute posterior mean and variance
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(
            x_0_pred, x_t, t
        )

        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, mask=None):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)

        Args:
            model: Denoising model
            x_t: Noisy data at timestep t
            t: Timesteps
            mask: Optional mask

        Returns:
            x_{t-1}
        """
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, mask)

        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, 1, 1)

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, mask=None, progress=True):
        """
        Generate samples using full denoising loop.

        Args:
            model: Denoising model
            shape: Shape of samples to generate (batch, seq_len, dim)
            mask: Optional mask
            progress: Whether to show progress bar

        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        timesteps = list(range(self.num_timesteps))[::-1]

        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, mask)

        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, ddim_steps=50, eta=0.0, mask=None, progress=True):
        """
        DDIM sampling for faster generation.

        Args:
            model: Denoising model
            shape: Shape of samples to generate
            ddim_steps: Number of sampling steps (< num_timesteps for speedup)
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
            mask: Optional mask
            progress: Whether to show progress bar

        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Create subsequence of timesteps
        c = self.num_timesteps // ddim_steps
        timesteps = np.asarray(list(range(0, self.num_timesteps, c))) + 1
        timesteps = timesteps[::-1]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        timesteps_iter = tqdm(timesteps, desc="DDIM Sampling") if progress else timesteps

        for i, t in enumerate(timesteps_iter):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t_batch, mask)

            # Predict x_0
            x_0_pred = self.predict_start_from_noise(x, t_batch, noise_pred)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Get next timestep
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)

            alpha = self.alphas_cumprod[t]
            alpha_next = alpha_next.unsqueeze(-1).unsqueeze(-1)
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)

            sigma = (
                    eta * torch.sqrt((1 - alpha_next) / (1 - alpha)) *
                    torch.sqrt(1 - alpha / alpha_next)
            )

            # DDIM update
            noise = torch.randn_like(x) if eta > 0 else 0
            x = (
                    torch.sqrt(alpha_next) * x_0_pred +
                    torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred +
                    sigma * noise
            )

        return x
