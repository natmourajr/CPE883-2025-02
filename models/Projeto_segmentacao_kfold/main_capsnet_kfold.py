import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from rock_seg_loader.rock_dataset_multi_kfold import RockSegmentationDatasetMulti
from rock_seg_model.caps_rock_seg.model_caps import SegCapsNet
from rock_seg_model.caps_rock_seg.caps_config import CAPS_CONFIG
from metrics import dice_score, iou_score

# ===== Configurações =====
root_dir = r"C:\Users\vrodrigues\Documents\Python Scripts\Dataset"
num_epochs = 10
num_folds = 10
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "Results/CapsNet_kfold"
os.makedirs(results_dir, exist_ok=True)

# ===== Dataset completo (train+val) =====
full_dataset = RockSegmentationDatasetMulti(root_dir=root_dir, split="trainval")

# ===== Dataset de teste fixo =====
test_dataset = RockSegmentationDatasetMulti(root_dir=root_dir, split="test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===== K-Fold =====
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_dice_history = []  # Para armazenar Dice médio de cada fold

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"\n===== Fold {fold+1}/{num_folds} =====")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # ===== Modelo =====
    model = SegCapsNet(CAPS_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    val_dices_epoch = []

    for epoch in range(1, num_epochs+1):
        # --- Treino ---
        model.train()
        total_train_loss = 0
        for imgs, masks, _ in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch} [Treino]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)

        # --- Validação ---
        model.eval()
        dices, ious = [], []
        val_images, val_masks_true, val_masks_pred, val_names = [], [], [], []

        with torch.no_grad():
            for imgs, masks, filenames in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                dices.append(dice_score(preds, masks, CAPS_CONFIG["num_classes"]))
                ious.append(iou_score(preds, masks, CAPS_CONFIG["num_classes"]))

                # Armazena 3 primeiras imagens para plot
                if len(val_images) < 3:
                    val_images.extend(imgs.cpu())
                    val_masks_true.extend(masks.cpu())
                    val_masks_pred.extend(preds.cpu())
                    val_names.extend(filenames)

        val_dice = np.mean(dices)
        val_dices_epoch.append(val_dice)

        # --- Salvar predições com rastreabilidade ---
        fold_epoch_dir = os.path.join(results_dir, f"fold_{fold+1}", f"epoch_{epoch}")
        os.makedirs(fold_epoch_dir, exist_ok=True)
        for i in range(len(val_images)):
            fig, axes = plt.subplots(1, 3, figsize=(9,3))
            axes[0].imshow(val_images[i][0], cmap="gray")
            axes[0].set_title("Original")
            axes[1].imshow(val_masks_true[i], cmap="viridis")
            axes[1].set_title("Mask Real")
            axes[2].imshow(val_masks_pred[i], cmap="viridis")
            axes[2].set_title("Mask Predita")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            base_name = os.path.splitext(val_names[i])[0]
            plt.savefig(os.path.join(fold_epoch_dir, f"{base_name}.png"))
            plt.close(fig)

        print(f"Fold {fold+1} Epoch {epoch} - Val Dice: {val_dice:.4f}")

    fold_dice_history.append(val_dices_epoch)

# ===== Curva final Dice médio com barra de erro =====
fold_dice_history = np.array(fold_dice_history)  # shape: (num_folds, num_epochs)
mean_dice = np.mean(fold_dice_history, axis=0)
std_dice = np.std(fold_dice_history, axis=0)

plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), mean_dice, marker='o', label="Dice Médio")
plt.fill_between(range(1, num_epochs+1), mean_dice - std_dice, mean_dice + std_dice,
                 color='blue', alpha=0.2, label="Std (barra de erro)")
plt.xlabel("Época")
plt.ylabel("Dice Score")
plt.title(f"Dice Médio de Validação - {num_folds}-Fold Cross Validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Dice_kfold.png"))
plt.show()
