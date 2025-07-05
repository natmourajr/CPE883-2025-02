from video_dataset import VideoDataset
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

"""
Preparação para prototipagem de modelos de classificação de vídeo.
Funcionalidades:
Leitura do vídeo - Abertura dos arquivos .mp4 e extração de quadros (frames)
Amostragem de quadros - Seleção de N quadros por vídeo com frame_stride
Transformações - Redimensionamento e conversão para tensor (usando transform)
Normalização estrutural - Todos os vídeos retornam com shape [T, C, H, W]
Empacotamento em batches - Via DataLoader → gera batches [B, T, C, H, W]
Associação com rótulos - Permite supervisionar aprendizado com labels[idx]
"""
video_paths = ["videos/video1.mp4", "videos/video2.mp4"]
labels = [0, 1]

transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

dataset = VideoDataset(
    video_paths=video_paths,
    labels=labels,
    transform=transform,
    frames_per_video=16,
    frame_stride=2,
    backend='torchvision'  # ou 'torchcodec' quando disponível
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for videos, targets in loader:
    print(videos.shape)  # [batch, T, C, H, W]
    print(targets)
