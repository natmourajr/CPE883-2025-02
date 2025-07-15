# cli.py
import typer
import torch
from torch.utils.data import DataLoader
from rock_dataset_multi import RockSegmentationDatasetMulti   #from rock_seg_loader.rock_dataset_multi import RockSegmentationDatasetMulti
from model import SimpleCKANModel                             #from ckan_simple_experiment.model import SimpleCKANModel
from torch import nn, optim
import os

app = typer.Typer()

@app.command()
def train(
    data_dir: str = typer.Option(r"E:\back_up_DOUTORADO\4_CPE883_ML2_2p\2_Dataset\mini_dataset", help="Caminho para o dataset"),
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RockSegmentationDatasetMulti(data_dir, output_shape=(128, 128))  # menor p/ treino rápido
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCKANModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (imgs, masks, _) in enumerate(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ▶️ Imprime as predições no primeiro batch da primeira época
            if epoch == 0 and batch_idx == 0:
                preds_class = preds.argmax(dim=1)
                print("▶ Predições:", preds_class.tolist())
                print("✔ Rótulos   :", masks.tolist())

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


if __name__ == "__main__":
    app()
