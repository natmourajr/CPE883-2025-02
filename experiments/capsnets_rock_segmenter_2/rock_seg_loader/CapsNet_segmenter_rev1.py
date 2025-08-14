import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi_rev1 import RockSegmentationDatasetMulti, split_dataset
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Usa backend não-interativo compatível com servidores/headless
import matplotlib.pyplot as plt
import os

# ==== Métricas ====
def dice_score(pred, target, num_classes):
    dice = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dice.append(2.0 * inter / union)
    return np.mean(dice)

def iou_score(pred, target, num_classes):
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - inter
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious)

# ==== Blocos CapsNet ====
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, caps_channels, caps_dim, kernel_size, stride):
        super().__init__()
        self.caps_dim = caps_dim
        self.caps_channels = caps_channels
        self.conv = nn.Conv2d(in_channels, caps_channels * caps_dim, kernel_size, stride)

    def forward(self, x):
        batch = x.size(0)
        out = self.conv(x)
        out = out.view(batch, self.caps_channels, self.caps_dim, out.size(2), out.size(3))
        out = self.squash(out)
        return out

    def squash(self, s):
        norm = torch.norm(s, dim=2, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))

class SegCapsNet(nn.Module):
    def __init__(self, img_channels=1, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=5, stride=1, padding=2)
        self.primary_caps = PrimaryCaps(32, caps_channels=8, caps_dim=8, kernel_size=1, stride=1)
        # stride=2 removido para não dobrar tamanho
        self.deconv = nn.ConvTranspose2d(8*8, 32, kernel_size=1, stride=1)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        caps = self.primary_caps(x)
        caps = caps.view(x.size(0), -1, caps.size(3), caps.size(4))  # [B, C*dim, H, W]
        x = F.relu(self.deconv(caps))
        out = self.final(x)
        return out
    
# ==== Função para salvar predições ====
def save_predictions(images, masks_true, masks_pred, epoch, save_dir="results"):
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
        plt.savefig(f"{save_dir}/epoch_{epoch}/sample_{i}.png")
        plt.close(fig)

# ==== Configuração ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RockSegmentationDatasetMulti(
    root_dir=r"C:\Users\vrodrigues\Documents\Python Scripts\mini_dataset",  # <-- mini conjunto
    output_shape=(128, 128)
)

#train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
#Utiliza os dados separados em treino, validação e teste
train_set, val_set, test_set = split_dataset(dataset)  # 70/15/15
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False)

model = SegCapsNet(img_channels=1, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ==== Loop de treino rápido ====
for epoch in range(3):  # poucas épocas só para ver se funciona
    model.train()
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    # Validação
    model.eval()
    dices, ious = [], []
    val_images, val_masks_true, val_masks_pred = [], [], []
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            dices.append(dice_score(preds, masks, num_classes=3))
            ious.append(iou_score(preds, masks, num_classes=3))
            # Salva alguns exemplos (até 3 por época)
            if len(val_images) < 3:
                val_images.extend(imgs.cpu())
                val_masks_true.extend(masks.cpu())
                val_masks_pred.extend(preds.cpu())

    print(f"Epoch {epoch+1}: Dice={np.mean(dices):.4f} IoU={np.mean(ious):.4f}")
    save_predictions(val_images, val_masks_true, val_masks_pred, epoch+1)
