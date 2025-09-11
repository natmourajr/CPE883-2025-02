#main_caps_v3.py
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi_rev2 import RockSegmentationDatasetMulti, split_dataset
from rock_seg_model.caps_rock_seg.model_caps import SegCapsNet    # <- trocar para o modelo desejado
from rock_seg_model.caps_rock_seg.caps_config import CAPS_CONFIG  # <- trocar para o config do modelo
from metrics import dice_score, iou_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# ===== Função para plotar matriz de confusão =====
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ===== Função para salvar predições =====
def save_predictions(images, masks_true, masks_pred, filenames, epoch, save_dir):
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

        base_name = os.path.splitext(filenames[i])[0]
        fig.suptitle(base_name, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_dir}/epoch_{epoch}/{base_name}.png")
        plt.close(fig)

# ===== Dataset =====
dataset = RockSegmentationDatasetMulti(
    root_dir=r"C:\Users\vrodrigues\Documents\Python Scripts\Dataset",
    output_shape=CAPS_CONFIG["output_shape"]
)
train_set, val_set, test_set = split_dataset(dataset)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=8, shuffle=False)

# ===== Modelo =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegCapsNet(CAPS_CONFIG).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# ===== Logs =====
train_losses, val_losses, test_losses = [], [], []
val_dices, val_ious = [], []

results_dir = os.path.join("Results", "CapsNet")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "results.txt")

with open(results_file, "w") as f:
    f.write("epoch | train_loss | val_loss | test_loss | val_Dice | val_IoU | test_Dice | test_IoU\n")


# ===== Loop de treino =====
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    # --- treino ---
    model.train()
    total_train_loss = 0
    for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Treino]", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)

    # --- validação ---
    model.eval()
    total_val_loss, dices, ious = 0, [], []
    val_images, val_masks_true, val_masks_pred, val_names = [], [], [], []
    all_true_pixels, all_pred_pixels = [], []
    with torch.no_grad():
        for imgs, masks, filename in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            dices.append(dice_score(preds, masks, CAPS_CONFIG["num_classes"]))
            ious.append(iou_score(preds, masks, CAPS_CONFIG["num_classes"]))

            # Para confusão: acumula pixels
            all_true_pixels.append(masks.cpu().numpy().flatten())
            all_pred_pixels.append(preds.cpu().numpy().flatten())

            if len(val_images) < 3:
                val_images.extend(imgs.cpu())
                val_masks_true.extend(masks.cpu())
                val_masks_pred.extend(preds.cpu())
                val_names.extend(filename)
    val_loss = total_val_loss / len(val_loader)
    val_dice, val_iou = np.mean(dices), np.mean(ious)

    # Salva predições
    save_predictions(val_images, val_masks_true, val_masks_pred, val_names,
                     epoch, save_dir=f"{results_dir}/val_set")
    
    # Matriz de confusão
    y_true = np.concatenate(all_true_pixels)
    y_pred = np.concatenate(all_pred_pixels)
    plot_confusion_matrix(y_true, y_pred, classes=['Fundo','Rocha','Poro'],
                          save_path=f"{results_dir}/val_set/confusion_matrix_epoch{epoch}.png")

    # --- teste ---
    total_test_loss, test_dices, test_ious = 0, [], []
    test_images, test_masks_true, test_masks_pred, test_names = [], [], [], []
    all_true_pixels, all_pred_pixels = [], []

    with torch.no_grad():
        for imgs, masks, filename in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_dices.append(dice_score(preds, masks, CAPS_CONFIG["num_classes"]))
            test_ious.append(iou_score(preds, masks, CAPS_CONFIG["num_classes"]))

            # Para confusão
            all_true_pixels.append(masks.cpu().numpy().flatten())
            all_pred_pixels.append(preds.cpu().numpy().flatten())

            if len(test_images) < 3:
                test_images.extend(imgs.cpu())
                test_masks_true.extend(masks.cpu())
                test_masks_pred.extend(preds.cpu())
                test_names.extend(filename)
    test_loss = total_test_loss / len(test_loader)
    test_dice, test_iou = np.mean(test_dices), np.mean(test_ious)

    # Salva predições
    save_predictions(test_images, test_masks_true, test_masks_pred, test_names,
                     epoch, save_dir=f"{results_dir}/test_set")
    
    # Matriz de confusão
    y_true = np.concatenate(all_true_pixels)
    y_pred = np.concatenate(all_pred_pixels)
    plot_confusion_matrix(y_true, y_pred, classes=['Fundo','Rocha','Poro'],
                          save_path=f"{results_dir}/test_set/confusion_matrix_epoch{epoch}.png")

    # --- logs ---
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    val_dices.append(val_dice)
    val_ious.append(val_iou)
    test_dices.append(test_dice)
    test_ious.append(test_iou)

    with open(results_file, "a") as f:
        f.write(f"{epoch} | {train_loss:.4f} | {val_loss:.4f} | {test_loss:.4f} | "
                f"{val_dice:.4f} | {val_iou:.4f} | {test_dice:.4f} | {test_iou:.4f}\n")

    tqdm.write(f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}, "
            f"val_Dice={val_dice:.4f}, val_IoU={val_iou:.4f}, "
            f"test_Dice={test_dice:.4f}, test_IoU={test_iou:.4f}")



# ===== Curvas de perda =====
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Curvas de perda")
plt.xlabel("Época"); plt.ylabel("Loss")
plt.grid(True); plt.tight_layout()
plt.savefig(f"{results_dir}/loss_curve.png")
plt.close()

# ===== Curvas DICE/IOU =====
plt.plot(val_dices, label="Val Dice", color="blue")
plt.plot(val_ious, label="Val IoU", color="cyan")
plt.plot(test_dices, label="Test Dice", color="red")
plt.plot(test_ious, label="Test IoU", color="orange")
plt.legend()
plt.title("Curvas de métricas")
plt.xlabel("Época"); plt.ylabel("Score")
plt.grid(True); plt.tight_layout()
plt.savefig(f"{results_dir}/DICE_IOU_curve.png")
plt.close()

