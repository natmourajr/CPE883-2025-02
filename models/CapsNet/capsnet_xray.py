import torch
from torch import nn
import torch.nn.functional as F

from .capsnet.capsulelayers import squash, DenseCapsule, PrimaryCapsule

class CapsNet(nn.Module):
    """
    Versão final da CapsNet, totalmente configurável via dicionário config
    e com cálculo dinâmico de shape para se adaptar a diferentes inputs.
    """
    def __init__(self, model_config, num_classes=2, device="cpu"):
        super(CapsNet, self).__init__()
        
        # Pega as seções corretas do dicionário de configuração
        self.image_size = model_config['preprocessing']['image_size']
        arch_config = model_config['architectures']['CapsNet']
        self.num_classes = num_classes
        print("Inicializando modelo CapsNet (Arquitetura Leve e Configurável)...")

        # --- Frontend Convolucional (lê do config) ---
        frontend_channels = arch_config['frontend_channels']
        self.conv1 = nn.Conv2d(in_channels=frontend_channels[0], out_channels=frontend_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=frontend_channels[1], out_channels=frontend_channels[2], kernel_size=3, padding=1)
        
        # --- Cápsulas Primárias (lê do config) ---
        self.primary_caps = PrimaryCapsule(
            in_channels=frontend_channels[2],
            out_channels=arch_config['primary_caps_out_channels'],
            dim_caps=arch_config['primary_caps_dim'],
            kernel_size=arch_config['primary_caps_kernel_size'],
            stride=2
        )
        
        # --- Cálculo Dinâmico do Tamanho das Cápsulas ---
        in_num_caps = self._get_primary_caps_output_size()
        
        # --- Cápsulas de Classe (lê do config) ---
        self.digit_caps = DenseCapsule(
            in_num_caps=in_num_caps,
            in_dim_caps=arch_config['primary_caps_dim'],
            out_num_caps=num_classes,
            out_dim_caps=arch_config['digit_caps_dim'],
            routings=arch_config['routings']
        )

        # --- Decoder (lê do config) ---
        self.decoder = nn.Sequential(
            nn.Linear(arch_config['digit_caps_dim'] * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.image_size * self.image_size),
            nn.Sigmoid()
        )

    def _get_primary_caps_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.image_size, self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.primary_caps(x)
            print(f"Número de cápsulas primárias calculado com a nova arquitetura: {x.size(1)}")
            return x.size(1)

    def forward(self, x, y_one_hot=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.primary_caps(x)
        digit_caps_output = self.digit_caps(x)
        y_pred = digit_caps_output.norm(dim=-1)

        if not self.training and y_one_hot is None:
            _, max_length_indices = y_pred.max(dim=1)
            y_one_hot = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
        
        if y_one_hot is not None:
             reconstruction_input = (digit_caps_output * y_one_hot[:, :, None]).view(digit_caps_output.size(0), -1)
             reconstruction = self.decoder(reconstruction_input)
             reconstruction = reconstruction.view(-1, 3, self.image_size, self.image_size)
        else:
            # Se y_one_hot for None mesmo no treino, não pode fazer a reconstrução.
            # Retorna um tensor vazio para compatibilidade.
            reconstruction = torch.zeros_like(x)

        return y_pred, reconstruction

class CapsNetStrided(nn.Module):
    """
    Versão da CapsNet que usa Convolução com Stride em vez de MaxPooling
    para reduzir a dimensionalidade.
    """
    def __init__(self, model_config, num_classes=2, device="cpu"):
        super(CapsNetStrided, self).__init__()
        
        self.image_size = model_config['preprocessing']['image_size']
        arch_config = model_config['architectures']['CapsNet']
        self.num_classes = num_classes
        print("Inicializando modelo CapsNet (com Strided Convolution)...")

        # --- Frontend com Stride ---
        frontend_channels = arch_config['frontend_channels']
        # A primeira convolução já reduz a imagem pela metade (stride=2)
        self.conv1 = nn.Conv2d(in_channels=frontend_channels[0], out_channels=frontend_channels[1], kernel_size=3, stride=2, padding=1)
        # A segunda convolução mantém o tamanho
        self.conv2 = nn.Conv2d(in_channels=frontend_channels[1], out_channels=frontend_channels[2], kernel_size=3, stride=1, padding=1)
        
        # --- Camadas de Cápsulas ---
        # A primary_caps faz a segunda redução de dimensionalidade (stride=2)
        self.primary_caps = PrimaryCapsule(
            in_channels=frontend_channels[2],
            out_channels=arch_config['primary_caps_out_channels'],
            dim_caps=arch_config['primary_caps_dim'],
            kernel_size=arch_config['primary_caps_kernel_size'],
            stride=2
        )
        
        # A beleza do nosso design: o cálculo dinâmico se adapta automaticamente!
        in_num_caps = self._get_primary_caps_output_size()
        
        self.digit_caps = DenseCapsule(
            in_num_caps=in_num_caps,
            # ... (resto dos parâmetros é igual ao da outra CapsNet)
            in_dim_caps=arch_config['primary_caps_dim'],
            out_num_caps=num_classes,
            out_dim_caps=arch_config['digit_caps_dim'],
            routings=arch_config['routings']
        )
        self.decoder = nn.Sequential(
            nn.Linear(arch_config['digit_caps_dim'] * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.image_size * self.image_size),
            nn.Sigmoid()
        )
    def _get_primary_caps_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.image_size, self.image_size)
            # Passa pelo novo frontend sem pooling
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.primary_caps(x)
            print(f"Número de cápsulas (stride): {x.size(1)}")
            return x.size(1)

    def forward(self, x, y_one_hot=None):
        # O forward pass agora não tem as chamadas para self.pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.primary_caps(x)
        digit_caps_output = self.digit_caps(x)
        y_pred = digit_caps_output.norm(dim=-1)

        if not self.training and y_one_hot is None:
            _, max_length_indices = y_pred.max(dim=1)
            y_one_hot = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
        
        if y_one_hot is not None:
             reconstruction_input = (digit_caps_output * y_one_hot[:, :, None]).view(digit_caps_output.size(0), -1)
             reconstruction = self.decoder(reconstruction_input)
             reconstruction = reconstruction.view(-1, 3, self.image_size, self.image_size)
        else:
            # Se y_one_hot for None mesmo no treino, não pode fazer a reconstrução.
            # Retornaum tensor vazio para compatibilidade.
            reconstruction = torch.zeros_like(x)

        return y_pred, reconstruction
