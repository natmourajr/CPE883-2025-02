from .ode_jump_encoder import ODEJumpEncoder
import torch

class ODEJumpEncoderDiffusion(ODEJumpEncoder):

    def __init__(self,denoiser,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser

    def test_model_preforward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        z = self.denoiser.denoise(
            state=self.denoiser.encoder(torch.cat([x, mask], dim=-1)),
            timestamps=timestamps,
            static_feats=static_feats,
            device=x.device,
            steps=self.denoise_steps if hasattr(self, 'denoise_steps') else 100
            )
        return self.denoiser.decoder(z)
        