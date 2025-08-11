import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 1. Baixar MNIST e salvar imagens em disco
def export_mnist_to_folder(folder='./mnist_images'):
    os.makedirs(folder, exist_ok=True)
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    for i, (img, label) in enumerate(mnist):
        if i >= 1000:  # apenas 1000 imagens
            break
        path = os.path.join(folder, f"{i}_{label}.png")
        img.save(path)
    print(f"Exported {i+1} MNIST images to {folder}")

# 2. Criar modelo e objeto GaussianDiffusion
def create_model_diffusion():
    model = Unet(
        dim=32,
        dim_mults=(1, 2, 4),
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=250,                #reduzido para acelerar
        sampling_timesteps=25,        #reduzido para acelerar
        objective='pred_v'
    )

    # Manda para GPU (ou CPU se não disponível)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)

    return diffusion

# 3. Executar treino com Trainer
def train():
    # Exporta imagens MNIST para pasta
    export_mnist_to_folder()

    diffusion = create_model_diffusion()

    trainer = Trainer(
        diffusion,
        folder='./mnist_images',
        train_batch_size=32,               #64
        train_lr=8e-5,
        train_num_steps=1000,              #5000
        save_and_sample_every=250,         #500
        results_folder='./results_mnist',
        calculate_fid=False                #Desativa o FID para economizar tempo e processamento
    )

    
    import signal
    import sys

    def handle_interrupt(sig, frame):
        print("\nTreinamento interrompido manualmente. Salvando progresso...")
        trainer.save("interrupted_model.pt")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    trainer.train()
        
if __name__ == '__main__':
    train()
