from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuse.models.components.utils import unsqueeze_as


class DiffusionProcess(nn.Module):
    def __init__(self, nb_timesteps: int = 50, start: float = 1e-4, end: float = 0.05):
        """
        Implements the diffusion process presented in DDPM.
        No learnable parameters, but relies on a trained 'score' model
        for sampling.
        :param nb_timesteps:
        :param start:
        :param end:
        """
        super().__init__()

        self.nb_timesteps = nb_timesteps

        # beta = likelihood variance q(x_t | x_t-1)
        beta = torch.linspace(start, end, nb_timesteps)
        alpha = 1.0 - beta
        alpha_hat = alpha.cumprod(dim=0)

        # variance of conditional prior, q(x_t|x_0)
        # = N(x_t ; sqrt(alpha_hat) * x_0, prior_variance)
        prior_variance = 1.0 - alpha_hat

        # forward process posterior variance (beta_hat) corresponding to q(x_t-1 | x_t, x_0)
        alpha_hat_t_1 = F.pad(alpha_hat, (1, 0))[:-1]
        posterior_variance = (1 - alpha_hat_t_1) * beta / (1 - alpha_hat)
        posterior_variance[0] = beta[0]

        for (name, tensor) in [
            ("beta", beta),
            ("alpha", alpha),
            ("alpha_hat", alpha_hat),
            ("prior_variance", prior_variance),
            ("posterior_variance", posterior_variance),
        ]:
            self.register_buffer(name, tensor)

    def sample_t(self, batch_size=1, device=None):
        """
        Sample a random time-step for each batch item.
        """
        return torch.randint(0, self.nb_timesteps, size=(batch_size,), device=device)

    def sample_q(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        """
        The "forward process". Given the data point x_0, we can sample
        any latent x_t from q(x_t|x_0)
        :param x0: the initial data point (batch, *)
        :param eps: noise samples from N(0, I) (batch, *)
        :param t: the timesteps in [0, nb_timesteps] (batch)
        """
        assert (t >= 0).all() and (t < self.nb_timesteps).all(), "Invalid time step"

        alpha_hat_t = unsqueeze_as(self.alpha_hat[t], x0.shape)  # (batch)
        return alpha_hat_t.sqrt() * x0 + (1.0 - alpha_hat_t).sqrt() * eps

    def sample_p(self, x_t: torch.Tensor, eps_hat: torch.Tensor, t: torch.Tensor, greedy: bool = False):
        """
        The "reverse process". Given a latent `x_t`, draw a sample from p(x_t-1|x_t)
        using the noise prediction.
        :param x_t: the previous sample (batch, *)
        :param eps_hat: the noise, predicted by neural net (batch, *)
        :param t: the time step (batch)
        :param greedy: use the mean
        """

        alpha_t = unsqueeze_as(self.alpha[t], x_t.shape)
        beta_t = unsqueeze_as(self.beta[t], x_t.shape)
        alpha_hat_t = unsqueeze_as(self.alpha_hat[t], x_t.shape)

        # calculate the mean
        mu = x_t - ((beta_t * eps_hat) / (1.0 - alpha_hat_t).sqrt())
        mu = (1.0 / alpha_t.sqrt()) * mu

        if greedy:
            return mu

        # sample
        std = unsqueeze_as(self.posterior_variance[t].sqrt(), x_t.shape)
        x_next = mu + std * torch.randn_like(mu)

        return x_next

    @torch.no_grad()
    def generate(
        self, shape: Tuple[int, ...], score_model: nn.Module, return_freq: int = 20
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of samples. Returns intermediate values as well.
        :param shape: a tuple indicating the shape of the data
        :param score_model: trained pytorch model, which produces the update
        :param return_freq: how often to accumulate intermediate values
        """
        device = list(score_model.parameters())[0].device
        batch, *_ = shape
        x_T = torch.randn(shape, device=device)

        x = x_T

        out = [x_T.cpu()]
        for t in range(self.nb_timesteps - 1, -1, -1):
            t_ = torch.full((batch,), t, dtype=torch.long, device=device)

            eps_hat = score_model(x, t_)
            x = self.sample_p(x, eps_hat, t_, greedy=t == 0)

            if (t + 1) % return_freq == 0:
                out.append(x.cpu())

        return x, out
