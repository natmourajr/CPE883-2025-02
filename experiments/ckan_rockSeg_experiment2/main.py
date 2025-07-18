# cli.py
import typer
import torch
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi import RockSegmentationDatasetMulti
from rock_seg_experiment2.model import CKANSegmentationModel
from torch import nn, optim
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

app = typer.Typer()

@app.command()
def train(
    data_dir: str = typer.Option(r"E:\back_up_DOUTORADO\4_CPE883_ML2_2p\2_Dataset\mini_dataset", help="Caminho para o dataset"),
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RockSegmentationDatasetMulti(data_dir, output_shape=(128, 128))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CKANSegmentationModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_pixels = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)  # [B, 1, H, W], [B, H, W]
            preds = model(imgs)  # [B, 3, H, W]
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred_labels = preds.argmax(dim=1)  # [B, H, W]
                correct += (pred_labels == masks).sum().item()
                total_pixels += masks.numel()

            pbar.set_postfix(loss=loss.item())

        acc = correct / total_pixels
        print(f"✅ Epoch {epoch+1}: Loss = {total_loss:.4f}, Acc = {acc:.4f}")

        # Exibe predições de debug
        visualize_predictions(imgs.cpu(), masks.cpu(), pred_labels.cpu())

def visualize_predictions(images, masks, preds):
    """Mostra a primeira imagem, máscara e predição do batch."""
    img = images[0][0]  # [H, W]
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
