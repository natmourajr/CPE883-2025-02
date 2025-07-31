# https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning?section=featured&tab=overview

import lightning as L
import diffusers
import torch
from diffusers.schedulers import DDPMScheduler


class DiffusionModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 scheduler: DDPMScheduler | None = None):
        super().__init__()
        self.model = model
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = DDPMScheduler()

    def training_step(self, x):
        noise = torch.randn_like(x)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        noisy_x = self.scheduler.add_noise(x, noise, steps)
        residual = self.model(noisy_x)
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def generate(self,
                 num_samples: int,
                 timesteps: int,
                 scheduler: DDPMScheduler | None = None):
        if scheduler is None:
            scheduler = DDPMScheduler()
        pipe = diffusers.DDPMPipeline(self.model, scheduler)
        (generated, ) = pipe(num_samples=num_samples,
                             timesteps=timesteps,
                             output_type='np.ndarray',
                             return_dict=False)
        return generated
