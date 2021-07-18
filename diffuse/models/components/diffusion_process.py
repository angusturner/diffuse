import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuse.models.components.utils import unsqueeze_as


class DiffusionProcess(nn.Module):
    def __init__(self, nb_timesteps=50, start=1e-4, end=0.05):
        """
        Defines the DDPM diffusion process
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

    def sample_q(self, x0, eps, t):
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

    def sample_p(self, x_t, eps_hat, t, greedy=False):
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
