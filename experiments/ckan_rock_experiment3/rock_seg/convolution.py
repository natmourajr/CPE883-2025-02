# rock_seg/convolution.py
import torch
import torch.nn.functional as F

def multiple_convs_kan_conv2d(x, convs, kernel_size, out_channels, stride, dilation, padding, device):
    B, C_in, H, W = x.shape
    x_unf = F.unfold(
        x,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation
    )  # [B, C_in * K*K, L]

    L = x_unf.shape[-1]
    x_unf = x_unf.transpose(1, 2)  # [B, L, C_in * K*K]
    x_unf = x_unf.reshape(B * L, -1, 1)  # [B*L, C_in*K*K, 1]

    outputs = []
    for i in range(out_channels):
        out_i = []
        for j in range(C_in):
            idx = i * C_in + j
            conv = convs[idx]
            x_slice = x_unf[:, j * kernel_size**2:(j + 1) * kernel_size**2, :].squeeze(-1)  # [B*L, K*K]
            y = conv.conv(x_slice) [0].squeeze(-1)  # Remove dimensão extra do KANLayer [B*L, 1] → [B*L]. Somente o tensor de saída, ignorando coeficientes
            out_i.append(y)
        outputs.append(torch.stack(out_i, dim=1).sum(dim=1))  # Soma sobre os canais de entrada

    out = torch.stack(outputs, dim=1)  # [B*L, out_channels]
    out = out.reshape(B, L, out_channels).transpose(1, 2)  # [B, out_channels, L]

    # Cálculo do tamanho da saída
    output_H = (H + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) // stride[0] + 1
    output_W = (W + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) // stride[1] + 1

    out = out.reshape(B, out_channels, output_H, output_W)  # [B, C_out, H_out, W_out]
    return out
