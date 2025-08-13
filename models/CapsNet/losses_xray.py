import torch
from torch import nn

class CapsuleLoss(nn.Module):
    """
    Envolve a função de perda da CapsNet (Margin Loss + Reconstruction Loss)
    em uma classe nn.Module para ser compatível com nosso pipeline.
    """
    def __init__(self, lam_recon=0.5):
        super(CapsuleLoss, self).__init__()
        self.lam_recon = lam_recon
        self.reconstruction_loss = nn.MSELoss(reduction='mean') # Como no paper original

    def forward(self, y_true, y_pred, x, x_recon):
        """
        Args:
            y_true (Tensor): Rótulos verdadeiros (one-hot encoded), shape=[batch, classes]
            y_pred (Tensor): Saída da CapsNet (comprimento dos vetores), shape=[batch, classes]
            x (Tensor): Imagens de entrada originais.
            x_recon (Tensor): Imagens reconstruídas pelo decoder.
        """
        # Margin Loss
        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        # Reconstruction Loss
        L_recon = self.reconstruction_loss(x_recon, x)

        # Perda Total (note a escala da perda de reconstrução)
        return L_margin + self.lam_recon * L_recon / x.size(0)