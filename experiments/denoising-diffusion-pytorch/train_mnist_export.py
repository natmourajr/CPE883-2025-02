#Arquivo principal
import os
from torchvision import datasets, transforms
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 1. Baixar MNIST e salvar imagens em disco
def export_mnist_to_folder(folder='./mnist_images'):
    os.makedirs(folder, exist_ok=True)

    mnist = datasets.MNIST(root='./data', train=True, download=True)

    for i, (img, label) in enumerate(mnist):
        # img já é PIL.Image.Image, não precisa converter
        path = os.path.join(folder, f"{i}_{label}.png")
        img.save(path)

    print(f"Exported {len(mnist)} MNIST images to {folder}")

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
        timesteps=1000,
        sampling_timesteps=100,
        objective='pred_v'
    )
    return diffusion

# 3. Executar treino com Trainer
def train():
    # Exporta imagens MNIST para pasta
    export_mnist_to_folder()

    diffusion = create_model_diffusion()

    trainer = Trainer(
        diffusion,
        folder='./mnist_images',
        train_batch_size=64,
        train_lr=8e-5,
        train_num_steps=5000,
        save_and_sample_every=500,
        results_folder='./results_mnist'
    )

    trainer.train()

if __name__ == '__main__':
    train()
