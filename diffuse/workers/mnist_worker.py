from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import wandb
from wandb.wandb_run import Run

from diffuse.datasets import MNIST
from diffuse.models import DiffusionProcess
from popgen.workers.abstract_worker import AbstractWorker


class MnistWorker(AbstractWorker):
    def __init__(
        self,
        exp_name: str,
        model: nn.Module,
        run_dir: str,
        run: Optional[Run],
        diffusion_settings: Dict,
        optim_class: str,
        optim_settings: Dict,
        *args,
        **kwargs,
    ):
        """
        :param exp_name:
        :param model:
        :param run_dir:
        :param wandb:
        :param diffusion_settings:
        :param optim_class:
        :param optim_settings:
        :param args:
        :param kwargs:
        """
        super(MnistWorker, self).__init__(exp_name, model, run_dir, run, *args, **kwargs)

        self.model = model

        # define the diffusion process
        self.diffusion = DiffusionProcess(**diffusion_settings)

        # setup the optimiser
        self.params = [p for p in model.parameters() if p.requires_grad]
        optim_class = getattr(torch.optim, optim_class)
        self.optim = optim_class(self.params, **optim_settings)

        # register the optimiser state, to include in the checkpoints
        self.register_state(self.optim, "optim")

        # put everything on GPU if available
        if torch.cuda.is_available():
            self.diffusion.cuda()
            self.cuda()

        # track the number of iterations
        self.iterations = {"train": 0, "test": 0}

        # load existing checkpoint
        self.load(checkpoint_id="best")

    # train / evaluation logic
    def main(self, loader, train=True):
        losses = []
        for i, (x0, _) in enumerate(tqdm(loader)):
            if train:
                self.optim.zero_grad()

            # put features on GPU
            x0 = x0.float().cuda()

            # sample x_t ~ q(x_t|x0), for a random step t ~ {0..nb_timesteps}
            eps = torch.randn_like(x0)
            t = self.diffusion.sample_t(eps.shape[0], device=eps.device)
            x_t = self.diffusion.sample_q(x0, eps, t)

            # predict the noise
            eps_hat = self.model(x_t, t)
            loss = F.mse_loss(eps_hat, eps, reduction="mean")

            # DEBUG - check for NaN values
            if torch.isnan(loss).any():
                raise Exception("NaN :(")

            losses.append(loss.item())

            if train:
                loss.backward()
                self.optim.step()

            self._plot_loss({"MSE": loss.item()}, train=train)

            if i % 500 == 0 and not self.train:
                self._plot_sample()

        return (np.mean(losses),)

    def train(self, loader):
        self.model.train()
        return self.main(loader, train=True)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        return self.main(loader, train=False)

    @torch.no_grad()
    def _plot_sample(self):
        x_gen, _ = self.diffusion.generate((1, 1, 28, 28), self.model)
        x_gen = MNIST.denormalize(x_gen).clamp(0, 1)
        x_np = x_gen.view(28, 28).cpu().numpy()
        self.wandb.log({"Forward Process Sample": wandb.Image(x_np)})
