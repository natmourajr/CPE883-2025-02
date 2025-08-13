import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rock_seg_loader.rock_dataset_multi import RockSegmentationDatasetMulti
import numpy as np

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

# ==== Configuração ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RockSegmentationDatasetMulti(
    root_dir=r"C:\Users\vrodrigues\Documents\Python Scripts\mini_dataset",  # <-- mini conjunto
    output_shape=(128, 128)
)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

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

    # Avaliação rápida
    model.eval()
    with torch.no_grad():
        dices, ious = [], []
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            dices.append(dice_score(preds, masks, num_classes=3))
            ious.append(iou_score(preds, masks, num_classes=3))
        print(f"Epoch {epoch+1}: Dice={np.mean(dices):.4f} IoU={np.mean(ious):.4f}")
