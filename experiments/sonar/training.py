import copy
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from alive_progress import alive_bar
import numpy as np
from sklearn.manifold import TSNE
import wandb
import skdim
from sklearn.manifold import trustworthiness
from skdim.id import KNN
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean, cosine


def calculate_local_step_consistency(series_embeddings: np.ndarray) -> float:
    """
    Measures the smoothness of the trajectory's velocity via Coefficient of Variation.
    A lower score is better (less jerky).
    """
    if len(series_embeddings) < 2:
        return np.nan
        
    # Calculate the length of each step between consecutive points
    step_lengths = [cosine(series_embeddings[i], series_embeddings[i+1]) 
                    for i in range(len(series_embeddings) - 1)]
    
    if not step_lengths or np.mean(step_lengths) == 0:
        return np.nan

    # Calculate Coefficient of Variation (CV) = std / mean
    mean_step = np.mean(step_lengths)
    std_step = np.std(step_lengths)
    
    return std_step / mean_step, step_lengths

def calculate_spearman_rank_correlation(series_embeddings: np.ndarray) -> float:
    """
    Measures if the trajectory is smoothly unfurling from its start point.
    A score close to +1.0 is best.
    """
    if len(series_embeddings) < 3:
        return np.nan # Cannot compute correlation with less than 3 points
        
    # Take the first point as the reference
    start_point = series_embeddings[0]
    
    # Calculate Euclidean distance from the start to all other points
    distances_from_start = [cosine(start_point, point) for point in series_embeddings[1:]]
    
    # The time ranks are just the order of the points
    time_ranks = np.arange(1, len(series_embeddings))
    
    # Calculate Spearman's rank correlation
    correlation, p_value = spearmanr(time_ranks, distances_from_start)
    
    return correlation, distances_from_start, time_ranks


def calculate_class_weights(y_train):
    # Compute class weights using sklearn's compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    # Convert the class weights to a PyTorch tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weights_tensor

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def sp_index(recall):
    return np.sqrt(recall.mean() * geo_mean(recall))


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, num_epochs=10, verbose=False, plotpath=None, wandb_logging=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.plotpath = plotpath
        self.wandb_logging = wandb_logging

    def train(self, train_loader, test_loader, patience=10):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            with alive_bar(len(train_loader), title=f"Training Epoch {epoch+1}/{self.num_epochs}") as bar:
                for batch_data, batch_target in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = self.criterion(output, batch_target)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    bar()
            
            val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = self.evaluate(test_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("=" * 80)
                    print(f"Early stopping at epoch {epoch+1}. Restoring best model state.")
                    print("=" * 80)
                    break
            
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = None
            
            if self.wandb_logging:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / len(train_loader),
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': np.mean(recall),
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'learning_rate': lr
                })

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {np.mean(recall):.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())
        
        val_loss /= len(test_loader)
        y_pred = np.array(all_preds)
        y_target = np.array(all_targets)
        
        accuracy = np.mean(recall_score(y_target, y_pred, average=None))
        precision = precision_score(y_target, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_target, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_target, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(
            label_binarize(y_target, classes=[0, 1, 2, 3]),
            label_binarize(y_pred, classes=[0, 1, 2, 3]),
            average='weighted',
            multi_class='ovr'
        )
        return val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target
    
    def evaluate_embeddings(self, test_loader, fig=None, ax=None, path=None):
        """
        Evaluates the embeddings using t-SNE for visualization.
        """
        self.model.eval()
        all_embeddings = []
        all_targets = []

        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                # Extract embeddings (before final output layer)
                embeddings = self.model(batch_data, embeddings=True)
                all_embeddings.append(embeddings.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())

        # Concatenate all embeddings into one array
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_targets = np.array(all_targets)
        
        print("Shape original:", all_embeddings.shape)

        num_samples = all_embeddings.shape[0]
        all_embeddings_flat = all_embeddings.reshape(num_samples, -1)
        
        print("Shape achatado:", all_embeddings_flat.shape)

        local_dim = skdim.id.TwoNN().fit_transform(all_embeddings_flat)
        intrinsic_dimension = local_dim.mean()

        # Apply t-SNE for dimensionality reduction (reduce to 2D)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings_flat)

        scores = {
            'trustworthiness-5':  trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=5),
            'trustworthiness-10': trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=10),
            'trustworthiness-20': trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=20),
            'intrinsic_dimension': intrinsic_dimension
        }

        for n_neighbors in [5, 10, 20, 30, 40, 50]:
            id_estimator = KNN(k=n_neighbors)
            id_estimator.fit(all_embeddings_flat)
            estimated_id = id_estimator.dimension_
            scores[f'knn-id-{n_neighbors}'] = estimated_id

        return embeddings_2d, all_targets, scores        

class DeepONetTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, num_epochs=10, verbose=False, plotpath=None, wandb_logging=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.plotpath = plotpath
        self.wandb_logging = wandb_logging

    def train(self, train_loader, test_loader, patience=10):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            with alive_bar(len(train_loader), title=f"Training Epoch {epoch+1}/{self.num_epochs}") as bar:
                for batch_data, batch_target, coords in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_data, coords)
                    loss = self.criterion(output, batch_target)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    bar()
            
            val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = self.evaluate(test_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("=" * 80)
                    print(f"Early stopping at epoch {epoch+1}. Restoring best model state.")
                    print("=" * 80)
                    break
            
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = None
            
            if self.wandb_logging:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / len(train_loader),
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': np.mean(recall),
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'learning_rate': lr
                })

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {np.mean(recall):.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target, coords in test_loader:
                output = self.model(batch_data, coords)
                loss = self.criterion(output, batch_target)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())
        
        val_loss /= len(test_loader)
        y_pred = np.array(all_preds)
        y_target = np.array(all_targets)
        
        accuracy = np.mean(recall_score(y_target, y_pred, average=None))
        precision = precision_score(y_target, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_target, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_target, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(
            label_binarize(y_target, classes=[0, 1, 2, 3]),
            label_binarize(y_pred, classes=[0, 1, 2, 3]),
            average='weighted',
            multi_class='ovr'
        )
        return val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target
    
    def evaluate_embeddings(self, test_loader, fig=None, ax=None, path=None):
        """
        Evaluates the embeddings using t-SNE for visualization.
        """
        self.model.eval()
        all_embeddings = []
        all_targets = []

        with torch.no_grad():
            for batch_data, batch_target, coords in test_loader:
                # Extract embeddings (before final output layer)
                embeddings = self.model(batch_data, coords, embeddings=True)
                all_embeddings.append(embeddings.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())

       # Concatenate all embeddings into one array
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_targets = np.array(all_targets)
        
        print("Shape original:", all_embeddings.shape)

        num_samples = all_embeddings.shape[0]
        all_embeddings_flat = all_embeddings.reshape(num_samples, -1)
        
        print("Shape achatado:", all_embeddings_flat.shape)

        local_dim = skdim.id.TwoNN().fit_transform(all_embeddings_flat)
        intrinsic_dimension = local_dim.mean()

        # Apply t-SNE for dimensionality reduction (reduce to 2D)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings_flat)

        scores = {
            'trustworthiness-5':  trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=5),
            'trustworthiness-10': trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=10),
            'trustworthiness-20': trustworthiness(all_embeddings_flat, embeddings_2d, n_neighbors=20),
            'intrinsic_dimension': intrinsic_dimension
        }

        for n_neighbors in [5, 10, 20, 30, 40, 50]:
            id_estimator = KNN(k=n_neighbors)
            id_estimator.fit(all_embeddings_flat)
            estimated_id = id_estimator.dimension_
            scores[f'knn-id-{n_neighbors}'] = estimated_id

        return embeddings_2d, all_targets, scores        

