# models/Room_Occupancy_Estimation/mlp_puro.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# MLP com Early Stopping

class MLP_Model(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim,
                 activation_fn="ReLU", output_activation=None,
                 dropout=0.2, use_batchnorm=False,
                 lr=1e-3, weight_decay=1e-5,
                 num_epochs=100, class_weights=None, patience=15):
        super(MLP_Model, self).__init__()

        # Mapeamento das funções de ativação
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "ELU": nn.ELU(),
            "GELU": nn.GELU()
        }
        self.activation_fn = activations.get(activation_fn, nn.ReLU())

        # Ativação da saída
        output_activations = {
            "Softmax": nn.Softmax(dim=1),
            "Sigmoid": nn.Sigmoid(),
            None: nn.Identity()
        }
        self.output_activation = output_activations.get(output_activation, nn.Identity())

        # Construção das camadas
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Parámetros de treino
        self.num_epochs = num_epochs
        self.lossList = []
        self.valLossList = []
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Função de perda com pesos de classes
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Otimizador
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)

        # Scheduler para reduzir LR quando não melhora
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='min', factor=0.5, patience=5
          )

    def forward(self, x):
        return self.output_activation(self.network(x))

    def fit(self, train_loader, val_loader, device):
        for epoch in range(self.num_epochs):
            # Treino
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            self.lossList.append(train_loss)

            # Avaliação
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self.forward(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            self.valLossList.append(val_loss)

            # Atualiza scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduzido de {old_lr:.6f} para {new_lr:.6f}")

            # Imprime o progresso
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Val loss did not improve for {self.patience} epochs.")
                    break

        # Carrega o melhor modelo salvo
        self.load_state_dict(torch.load('best_model.pth'))

    def predict(self, loader, device):
        self.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                outputs = self.forward(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_pred)

    def evaluate(self, loader, device):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_loss += loss.item()
        return running_loss / len(loader)