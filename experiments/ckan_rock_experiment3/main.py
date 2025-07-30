### main.py
import typer
import torch
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi import RockSegmentationDatasetMulti
from rock_seg.model import CKANSegmentationModel
from rock_seg.config import CKAN_CONFIG
from torch import nn, optim
import matplotlib
matplotlib.use('TkAgg')  # Usa backend não-interativo compatível com servidores/headless
import matplotlib.pyplot as plt
from tqdm import tqdm

app = typer.Typer()

@app.command()
def train(
    data_dir: str = r"C:\Users\vrodrigues\Documents\Python Scripts\mini_dataset_2",
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diagnóstico:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("📊 Diagnóstico de memória CUDA antes do carregamento do modelo/dados:")
        print(torch.cuda.memory_summary(device=device, abbreviated=False))

    dataset = RockSegmentationDatasetMulti(data_dir, output_shape=CKAN_CONFIG["output_shape"])  #dataset = RockSegmentationDatasetMulti(data_dir, output_shape=(128, 128))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CKANSegmentationModel().to(device)                                                  #model = CKANSegmentationModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        #print(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total_pixels = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks, _ in pbar:
            #print(f"Batch imgs shape: {imgs.shape}, masks shape: {masks.shape}")
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == masks).sum().item()
                total_pixels += masks.numel()

            pbar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(dataloader)
        losses.append(epoch_loss)
        acc = correct / total_pixels
        print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Acc = {acc:.4f}")
        visualize_predictions(imgs.cpu(), masks.cpu(), pred_labels.cpu())

    plt.plot(losses)
    plt.title("Curva de perda")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_predictions(images, masks, preds):
    img = images[0][0]
    mask = masks[0]
    pred = preds[0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Imagem")

    axs[1].imshow(mask, cmap='viridis')
    axs[1].set_title("Máscara Verdadeira")

    axs[2].imshow(pred, cmap='viridis')
    axs[2].set_title("Predição")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app()
