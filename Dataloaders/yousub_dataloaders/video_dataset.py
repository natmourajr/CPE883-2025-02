import os
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision.transforms import Compose
from typing import List, Callable, Optional, Union


class VideoDataset(Dataset):
    """
    A função VideoReader do torchvision.io está obsoleta e a PyTorch recomenda migrar para TorchCodec. Problemas com TorchCodec atualmente: A API 
    ainda está evoluindo. Isso significa que a integração com torch.utils.data.Dataset exige algum cuidado e adaptação.
    Estratégia: 
    Curto prazo (continuação de prototipagem):
    Continuar com o VideoReader no módulo video_dataset.py para testes iniciais, pois ele funciona bem em prototipagem e datasets 
    pequenos/médios; A migração para TorchCodec pode ser adiada para quando a API estiver mais estável.
    Longo prazo (preparação para produção):
    Refatorar a classe para permitir injeção de backend de leitura, por exemplo: backend='torchvision' ou backend='torchcodec'.
    
    Dataset customizado para vídeos, com suporte a torchvision (com VideoReader, para prototipagem rápida) e torchcodec (interface futura, com 
    base preparada para implementação posterior).

    Args:
        video_paths (List[str]): Lista com os caminhos dos vídeos.
        labels (Optional[List[Union[int, float]]]): Lista de rótulos (se houver).
        transform (Optional[Callable]): Transformações a serem aplicadas em cada quadro.
        frames_per_video (Optional[int]): Número de quadros a amostrar por vídeo.
        frame_stride (int): Intervalo entre quadros. Padrão = 1.
        backend (str): 'torchvision' (default) ou 'torchcodec' (experimental).
    """
    def __init__(
        self,
        video_paths: List[str],
        labels: Optional[List[Union[int, float]]] = None,
        transform: Optional[Callable] = None,
        frames_per_video: Optional[int] = None,
        frame_stride: int = 1,
        backend: str = "torchvision"
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.frame_stride = frame_stride
        self.backend = backend.lower()

        if self.labels is not None and len(self.labels) != len(self.video_paths):
            raise ValueError("O número de labels deve ser igual ao número de vídeos.")

        if self.backend not in ['torchvision', 'torchcodec']:
            raise ValueError("Backend inválido. Use 'torchvision' ou 'torchcodec'.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        if self.backend == 'torchvision':
            video_tensor = self._read_video_torchvision(video_path)
        elif self.backend == 'torchcodec':
            video_tensor = self._read_video_torchcodec(video_path)

        if self.labels is not None:
            return video_tensor, self.labels[idx]
        return video_tensor

    def _read_video_torchvision(self, path):
        reader = VideoReader(path, "video")
        reader.seek(0)

        frames = []
        count = 0

        for i, frame in enumerate(reader):
            if i % self.frame_stride != 0:
                continue
            img = frame['data']  # [H, W, C]
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            count += 1
            if self.frames_per_video is not None and count >= self.frames_per_video:
                break

        if not frames:
            raise RuntimeError(f"Nenhum quadro válido foi extraído de {path}.")
        return torch.stack(frames)

    def _read_video_torchcodec(self, path):
        raise NotImplementedError("Leitura com TorchCodec ainda não implementada. "
                                  "instalar e integrar com: pip install torchcodec")
