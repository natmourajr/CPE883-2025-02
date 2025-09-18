# models/Room_Occupancy_Estimation/mlp_deeponet.py
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

        # Parâmetros de treino
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

# Implementação da DeepONet

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_input_dim, trunk_hidden_dims, output_dim,
                 activation_fn="ReLU", dropout=0.2, use_batchnorm=False):
        super(DeepONet, self).__init__()

        # Branch Net
        self.branch_net = branch_net

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

        # Trunk Net
        trunk_layers = []
        prev_dim = trunk_input_dim

        for h_dim in trunk_hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                trunk_layers.append(nn.BatchNorm1d(h_dim))
            trunk_layers.append(self.activation_fn)
            trunk_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        trunk_output_dim = 64
        trunk_layers.append(nn.Linear(prev_dim, trunk_output_dim))

        self.trunk_net = nn.Sequential(*trunk_layers)

        self.final_layer = nn.Linear(trunk_output_dim, output_dim)

    def forward(self, branch_input, trunk_input):

        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        combined = branch_output * trunk_output
        output = self.final_layer(combined)

        return output

class DeepONetWrapper(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, branch_hidden_layers,
                 trunk_hidden_layers, output_dim, **kwargs):
        super(DeepONetWrapper, self).__init__()

        # Configurações de treinamento
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.window_size = kwargs.get('window_size', 25)
        self.features_size = kwargs.get('features_size', 5)

        # Filtra os parâmetros específicos para o MLP_Model
        mlp_kwargs = {
            'activation_fn': kwargs.get('activation_fn', 'ReLU'),
            'output_activation': None,
            'dropout': kwargs.get('dropout', 0.2),
            'use_batchnorm': kwargs.get('use_batchnorm', False),
            'lr': kwargs.get('lr', 1e-4),
            'weight_decay': kwargs.get('weight_decay', 1e-5),
            'num_epochs': kwargs.get('num_epochs', 50),
            'class_weights': None,
            'patience': kwargs.get('patience', 15)
        }

        # Cria a branch net (MLP normal)
        self.branch_net = MLP_Model(
            input_dim=branch_input_dim,
            hidden_layers=branch_hidden_layers,
            output_dim=64,
            **mlp_kwargs
        )

        # Remove o otimizador da branch net pois será treinada como parte da DeepONet
        if hasattr(self.branch_net, 'optimizer'):
            delattr(self.branch_net, 'optimizer')
        if hasattr(self.branch_net, 'scheduler'):
            delattr(self.branch_net, 'scheduler')

        # Cria a DeepONet completa
        self.deeponet = DeepONet(
            branch_net=self.branch_net,
            trunk_input_dim=trunk_input_dim,
            trunk_hidden_dims=trunk_hidden_layers,
            output_dim=output_dim,
            activation_fn=kwargs.get('activation_fn', 'ReLU'),
            dropout=kwargs.get('dropout', 0.2),
            use_batchnorm=kwargs.get('use_batchnorm', False)
        )

        # Função de perda
        if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
            self.criterion = nn.CrossEntropyLoss(weight=kwargs['class_weights'])
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.deeponet.parameters(),
                                          lr=kwargs.get('lr', 1e-4),
                                          weight_decay=kwargs.get('weight_decay', 1e-5))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.lossList = []
        self.valLossList = []

        # Adiciona Early Stopping no Wrapper também
        self.patience = kwargs.get('patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def forward(self, branch_input, trunk_input):
        return self.deeponet(branch_input, trunk_input)

    def prepare_inputs(self, X_batch, device):
        X_batch = X_batch.to(device)


        if X_batch.dim() == 3 and X_batch.shape[1] == self.window_size and X_batch.shape[2] == self.features_size:
            branch_input = X_batch.view(X_batch.size(0), -1)
            trunk_input = X_batch[:, -1, :]
            return branch_input, trunk_input

        total_features = X_batch.size(1)

        # Se temos features suficientes para a janela temporal
        if total_features >= self.window_size * self.features_size:
            # Pega as primeiras features para a branch net (série temporal)
            branch_input = X_batch[:, :self.window_size * self.features_size]
            # Pega as últimas features para a trunk net (último timestep)
            trunk_input = X_batch[:, -self.features_size:]
        else:
            # Estratégia alternativa para dados menores
            print(f"AVISO: Dados com apenas {total_features} features, esperado {self.window_size * self.features_size}")
            branch_input = X_batch
            trunk_input = X_batch[:, :min(self.features_size, total_features)]

        return branch_input, trunk_input

    def fit(self, train_loader, val_loader, device):
        for epoch in range(self.num_epochs):
            # Treino
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Prepara inputs
                branch_input, trunk_input = self.prepare_inputs(X_batch, device)

                # Verifica se as dimensões estão corretas
                if branch_input.size(1) != self.window_size * self.features_size:
                    print(f"Aviso: Dimensão do branch input {branch_input.shape} não corresponde ao esperado {self.window_size * self.features_size}")
                    continue

                if trunk_input.size(1) != self.features_size:
                    print(f"Aviso: Dimensão do trunk input {trunk_input.shape} não corresponde ao esperado {self.features_size}")
                    continue

                self.optimizer.zero_grad()
                outputs = self.forward(branch_input, trunk_input)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.lossList.append(train_loss)

            # Validação
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    branch_input, trunk_input = self.prepare_inputs(X_batch, device)

                    if (branch_input.size(1) != self.window_size * self.features_size or
                        trunk_input.size(1) != self.features_size):
                        continue

                    outputs = self.forward(branch_input, trunk_input)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader) if len(val_loader) > 0 else 0
            self.valLossList.append(val_loss)

            # Atualiza scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if new_lr != old_lr:
                    print(f"Learning rate reduzido de {old_lr:.6f} para {new_lr:.6f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Val loss did not improve for {self.patience} epochs.")
                    break