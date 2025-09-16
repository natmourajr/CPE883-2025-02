#main_caps_v1.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi_rev2 import RockSegmentationDatasetMulti, split_dataset
from rock_seg_model.caps_rock_seg.model_caps import SegCapsNet
from rock_seg_model.caps_rock_seg.caps_config import CAPS_CONFIG
from metrics import dice_score, iou_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# ===== Função unificada para salvar predições =====
def save_predictions(images, masks_true, masks_pred, filenames, epoch, save_dir="results_v1/CapsNet"):
    os.makedirs(f"{save_dir}/epoch_{epoch}", exist_ok=True)
    for i in range(len(images)):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(images[i][0].cpu(), cmap="gray")
        axes[0].set_title("Imagem Original")
        axes[1].imshow(masks_true[i].cpu(), cmap="viridis")
        axes[1].set_title("Máscara Real")
        axes[2].imshow(masks_pred[i].cpu(), cmap="viridis")
        axes[2].set_title("Máscara Predita")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        # Usa o nome original do arquivo (sem extensão .tif)
        base_name = os.path.splitext(filenames[i])[0]

        # Adiciona título com o nome do arquivo na parte superior da figura
        fig.suptitle(base_name, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # deixa espaço pro título
        plt.savefig(f"{save_dir}/epoch_{epoch}/{base_name}.png")
        plt.close(fig)

# ===== Dataset =====
dataset = RockSegmentationDatasetMulti(
    root_dir=r"C:\Users\vrodrigues\Documents\Python Scripts\mini_dataset",
    output_shape=CAPS_CONFIG["output_shape"]
)
train_set, val_set, test_set = split_dataset(dataset)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)

# ===== Modelo =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegCapsNet(CAPS_CONFIG).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# ===== Loop de Treino =====
losses = []
for epoch in range(5):  # Ajuste de número de épocas
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    epoch_loss = total_loss / len(train_loader)
    losses.append(epoch_loss)

    # ===== Validação =====
    model.eval()
    dices, ious = [], []
    val_images, val_masks_true, val_masks_pred, sample_names  = [], [], [], []
    with torch.no_grad():
        for imgs, masks, filename in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            dices.append(dice_score(preds, masks, CAPS_CONFIG["num_classes"]))
            ious.append(iou_score(preds, masks, CAPS_CONFIG["num_classes"]))
            if len(val_images) < 3:
                val_images.extend(imgs.cpu())
                val_masks_true.extend(masks.cpu())
                val_masks_pred.extend(preds.cpu())
                sample_names.extend(filename)  # acumula nomes    #sample_names = [fn for fn in filename]  # <- guarda nomes

    #print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Dice={np.mean(dices):.4f}, IoU={np.mean(ious):.4f}")
    tqdm.write(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Dice={np.mean(dices):.4f}, IoU={np.mean(ious):.4f}")
    save_predictions(val_images, val_masks_true, val_masks_pred, sample_names, epoch+1)

# ===== Salvar curva de loss =====
plt.plot(losses)
plt.title("Curva de perda")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results_v1/CapsNet/loss_curve.png")
plt.close()
