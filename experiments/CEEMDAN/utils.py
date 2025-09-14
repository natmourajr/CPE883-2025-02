
import torch
import torch.nn as nn

# 5. Training loop
def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        if test_loader is not None:

            # validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(test_loader))

        if test_loader is not None:   
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}")

    return train_losses, val_losses


class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mean_X=None, std_X=None, mean_y=None, std_y=None, fit=True):
        self.base_dataset = base_dataset

        if fit:  # calcula estatísticas a partir do dataset (treino)
            all_X = torch.cat([x for x, _ in base_dataset], dim=0)
            all_y = torch.cat([y for _, y in base_dataset], dim=0)

            self.mean_X = all_X.mean(dim=0, keepdim=True)
            self.std_X = all_X.std(dim=0, keepdim=True)

            self.mean_y = all_y.mean(dim=0, keepdim=True)
            self.std_y = all_y.std(dim=0, keepdim=True)
        else:  # usa estatísticas já calculadas no treino
            self.mean_X = mean_X
            self.std_X = std_X
            self.mean_y = mean_y
            self.std_y = std_y

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, y = self.base_dataset[idx]
        X = (X - self.mean_X) / (self.std_X + 1e-8)
        y = (y - self.mean_y) / (self.std_y + 1e-8)
        return X, y

