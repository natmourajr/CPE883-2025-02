import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np 
max_drop = 0.7
TS_SPAN = 60 * 60 * 24 * 365
from sklearn.model_selection import TimeSeriesSplit, StratifiedGroupKFold

# ---------- Helpers privados ----------
def _window_starts_and_count(n_rows: int, window_size: int, window_step: int):
    if window_size is None or window_size >= n_rows:
        starts = np.array([0], dtype=int)
        n_ws = 1
    else:
        starts = np.arange(0, n_rows - window_size + 1, window_step, dtype=int)
        n_ws = len(starts)
    return starts, n_ws

def _groups_non_overlap_from_starts(starts: np.ndarray, G: int):
    """Gera ids de grupo sem sobreposição (cada janela fica em um bloco de tamanho G)."""
    return (starts // G).astype(int)

def _states_from_df_windows(df: pd.DataFrame,
                            states_col: list,
                            window_size: int,
                            window_step: int,
                            label_at: str = "end"):
    """
    Retorna (y_win: (N_win,), starts: (N_win,)).
    Se states_col for one-hot/prob, usa argmax por timestamp.
    label_at='end' usa o último timestamp da janela; 'mode' usa a moda na janela.
    """
    n_rows = len(df)
    starts, n_ws = _window_starts_and_count(n_rows, window_size, window_step)
    # matriz (L, S) com scores/one-hot de estados
    S = df[states_col].to_numpy(dtype=np.float32)  # (L, S)
    if S.ndim != 2:
        raise ValueError("states_col deve produzir uma matriz (L, n_states).")
    n_states = S.shape[1]

    # índice de classe por timestamp (argmax ao longo dos estados)
    y_ts = np.argmax(S, axis=1).astype(int)  # (L,)

    y_win = np.empty(n_ws, dtype=int)
    if n_ws == 1:
        if label_at == "end":
            y_win[0] = y_ts[n_rows - 1]
        elif label_at == "mode":
            vals, cnts = np.unique(y_ts, return_counts=True)
            y_win[0] = vals[np.argmax(cnts)]
        else:
            raise ValueError("label_at deve ser 'end' ou 'mode'")
        return y_win, starts

    for k, s in enumerate(starts):
        e = s + window_size
        if label_at == "end":
            y_win[k] = y_ts[e - 1]
        elif label_at == "mode":
            vals, cnts = np.unique(y_ts[s:e], return_counts=True)
            y_win[k] = vals[np.argmax(cnts)]
        else:
            raise ValueError("label_at deve ser 'end' ou 'mode'")
    return y_win, starts

def _make_weighted_sampler_from_classes(y_classes: np.ndarray):
    """Peso inverso à frequência da classe (estado)."""
    binc = np.bincount(y_classes)
    binc = np.maximum(binc, 1)
    w_per_class = 1.0 / binc
    weights = w_per_class[y_classes]
    weights = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),   # um 'epoch' lógico
        replacement=True
    )
    return sampler

class ODEFunc(nn.Module):
    """f_θ usado no trecho contínuo  dh/dt = f_θ(h)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, h):          # t é obrigatório p/ torchdiffeq
        return self.net(h)
        
        
class JumpODE(nn.Module):
    """
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        
    def forward(self, x, ts):
        """
        x  : (B, T, C)
        ts : (B, T)   segundos unix (normalizados ou não)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []

        for i in range(T):
            if i > 0:
                dt_i = (ts[:, i] - ts[:, i-1]).float().unsqueeze(-1)  # (B,1)
                f1 = self.odefunc(h)
                f2 = self.odefunc(h + 0.5*dt_i*f1)
                f3 = self.odefunc(h + 0.5*dt_i*f2)
                f4 = self.odefunc(h + dt_i*f3)
                h  = h + (dt_i/6.0)*(f1 + 2*f2 + 2*f3 + f4)
            h = self.gru(x[:, i], h)                                 # jump
            states.append(h)

        H = torch.stack(states, dim=1)   # (B, T, hidden_dim)

        return H


class ODEJump(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        lam: list[float,float] = [0.9, 0.1]
        
    ):
        self.lam = lam
        super().__init__()
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )

        self.odejump = JumpODE(hidden_dim, hidden_dim)
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        # Embedding de entrada
        if not already_latent:
            h = self.encoder(torch.cat([x, mask], dim=-1))
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        h = self.odejump(h, timestamps)   # (B,T,model_dim)
        state = h
        return state,self.decoder(state) if return_x_hat else None

    def _compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor,
        mask_train: torch.Tensor
    ):
        #L1
        mask_err = mask * (1 - mask_train) # erro ao longo dos C canais observados 
        sse = ((x - x_hat)**2 * mask_err).sum(dim=-1) # (B,T) 
        nobs = mask_err.sum(dim=-1).clamp(min=1e-8) # -½ λ ||x-μ||^2 + ½ log λ
        L1 = sse.sum()
        # ----- L4 (máscara) -----
        # máscara binária: 1 se ao menos um canal está presente no timestep
                # (B, T, 1)
        m_t = mask_train.any(dim=2, keepdim=True).float()              # (B,T,1)
        mb_pred = torch.sigmoid(self.miss_head(state)).clamp(1e-4, 1-1e-4)  # (B, T, 1)
        L4 = F.binary_cross_entropy(mb_pred, m_t, reduction='sum')
        L1_div = nobs.sum().clamp(min=1.0)
        L4_div = float(mb_pred.numel())
        loss = self.lam[0]*L1/L1_div + self.lam[1]*L4/L4_div

        return (
            loss,
            (float(L1.item()), float(L1_div.item())),
            (float(L4.item()), float(L4_div))
            )
    
    # ---------- Novo método: test_model (macro/micro e por estado) ----------
    def test_model(self,
        loader: DataLoader,
        states_col: list,
        feature_cols: list,
        reduce: str = "mean"
        ):
        """
        Avaliação por janela:
        - MSE por classe (estado) com macro-média (cada estado pesa igual).
        - MSE micro (ponderado por nº de observações).
        A classe da janela é inferida de states_col (argmax no último timestamp da janela).
        """
        device = next(self.parameters()).device
        self.eval()
        per_class_sse = {}
        per_class_n = {}
        with torch.no_grad():
            for batch in loader:
                # batch = (x, ts, mask) ou (x, ts, mask, static)
                if len(batch) == 4:
                    x, ts_batch, m, s = batch
                else:
                    x, ts_batch, m = batch; s = None
                x = x.to(device, non_blocking=True)
                ts_batch = ts_batch.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                if s is not None: s = s.to(device, non_blocking=True)

                # Máscara de avaliação: observar só o que não foi mascarado no treino
                m_eval = m.clone()
                m_train = m.clone()
                m_train[:, -1, :] = 0.0  # mesmo critério do treino
                x_masked = x * m_train

                # forward
                state, x_hat = self.forward(x_masked, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m_train)

                # SSE por janela, considerando apenas observações m_eval*(1 - m_train)
                mask_err = m_eval * (1.0 - m_train)
                sse_bt = ((x - x_hat) ** 2 * mask_err).sum(dim=(1, 2))   # (B,)
                nobs_bt = mask_err.sum(dim=(1, 2)).clamp(min=1.0)        # (B,)

                # Classe da janela: inferida do próprio x (ou melhor, do df — aqui assumimos que
                # o usuário já incluiu states em feature_cols; se não, substitua por logits externos)
                # Se states NÂO estiverem em feature_cols, troque esta lógica para trazer o rótulo de fora.
                # Aqui: suponho que os estados (one-hot/prob) estejam concatenados ao final de feature_cols.
                # Se preferir, passe um tensor de labels junto no loader.
                # -> Para robustez, pego a "coluna" de estados como as últimas len(states_col).
                S_idx_start = x.shape[-1] - len(states_col)
                S_hat = x[:, :, S_idx_start:]  # (B, T, S) - os estados originais na janela
                y_bt = torch.argmax(S_hat[:, -1, :], dim=-1).detach().cpu().numpy()  # estado no último timestamp

                for b in range(x.shape[0]):
                    cls = int(y_bt[b])
                    per_class_sse[cls] = per_class_sse.get(cls, 0.0) + float(sse_bt[b].item())
                    per_class_n[cls]   = per_class_n.get(cls, 0.0)   + float(nobs_bt[b].item())

        # Agregações
        classes = sorted(per_class_n.keys())
        per_class_mse = {c: (per_class_sse[c] / max(per_class_n[c], 1.0)) for c in classes}
        # macro = média simples entre classes presentes
        macro_mse = float(np.mean([per_class_mse[c] for c in classes])) if classes else float("nan")
        # micro = soma(SSE)/soma(N)
        micro_mse = float(sum(per_class_sse.values()) / max(sum(per_class_n.values()), 1.0)) if classes else float("nan")

        return {"macro_mse": macro_mse, "micro_mse": micro_mse, "per_class_mse": per_class_mse}
    
    @staticmethod
    def _make_dataset(df, timestamp_col, window_size, feature_cols, static_features_cols, window_step=1):
        if timestamp_col != 'index':
            df = df.sort_values(timestamp_col).reset_index(drop=True)
# ---------------- NORMALIZAÇÃO DO TEMPO ----------------
        if timestamp_col != "index":
            ts_raw = pd.to_datetime(df[timestamp_col]).astype("int64") / 1e9
        else:
            ts_raw = pd.to_datetime(df.index).astype("int64") / 1e9
        t0 = ts_raw[0]
        ts_rel = ((ts_raw - t0) / TS_SPAN).to_numpy(dtype=np.float32)               # começa em 0

        times = torch.from_numpy(ts_rel)            # (L,)           # (L,)
        values_np = df[feature_cols].values   
        mask_np   = ~pd.isna(values_np) 
        values_np = np.nan_to_num(values_np, nan=0.0)
        data  = torch.tensor(values_np, dtype=torch.float32)
        mask  = torch.tensor(mask_np,  dtype=torch.float32)  # (L,C)
        #times = torch.tensor(ts.values, dtype=torch.float32)
        static = torch.tensor(df[static_features_cols].values, dtype=torch.float32) if static_features_cols else None
        if window_size is None or window_size >= len(df):
            seqs = data.unsqueeze(0)
            ts_seqs = times.unsqueeze(0)
            stat_seqs = static[0].unsqueeze(0) if static is not None else None
            mask_seqs = mask.unsqueeze(0)   # (1,L,C)
        else:
            n_ws = (len(df) - window_size) // window_step + 1
            seqs = torch.stack([data[i:i+window_size] for i in range(0,n_ws,window_step)])
            ts_seqs = torch.stack([times[i:i+window_size] for i in range(0,n_ws,window_step)])
            mask_seqs = torch.stack([mask[i:i+window_size] for i in range(0,n_ws,window_step)])
            stat_seqs = static[0].unsqueeze(0).repeat(n_ws, 1)  if static is not None else None
        if stat_seqs is None:
            return TensorDataset(seqs, ts_seqs, mask_seqs)                   # 3 itens
        return TensorDataset(seqs, ts_seqs, mask_seqs, stat_seqs)   

    # ---------- Novo método: train_cognite ----------
    def train_cognite(self,
                    df: pd.DataFrame,
                    feature_cols: list,
                    static_features_cols: list,
                    timestamp_col: str,
                    states_col: str,
                    batch_size: int = 32,
                    lr: float = 3e-4,
                    window_size: int = None,
                    window_step: int = 1,
                    epochs: int = 10,
                    validate: bool = True,
                    early_stopping: bool = True,
                    patience: int = 5,
                    device: torch.device = None,
                    n_splits_outer: int = 5,
                    n_splits_inner: int = 4,
                    label_at: str = "end",
                    seed_outer: int = 42,
                    seed_inner: int = 123):
        """
        Split com StratifiedGroupKFold:
        - Estratificação = estado por janela (derivado de states_col).
        - Grupos = blocos sem sobreposição de tamanho G=window_size para evitar vazamento.
        Oversampling: WeightedRandomSampler por classe (estado).
        Early stopping: melhor macro-MSE por estado no conjunto de validação (ou teste, se validate=False).
        """
        states_col = [states_col]
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_sorted = df if timestamp_col == "index" else df.sort_values(timestamp_col).reset_index(drop=True)

        # 1) Constrói Dataset de sequência/tempo/máscara/estático
        trainval_ds = self._make_dataset(
            df_sorted, timestamp_col, window_size, feature_cols, static_features_cols, window_step=window_step
        )
        # 2) Rótulos de estado por janela e grupos sem overlap
        y_win, starts = _states_from_df_windows(
            df_sorted, states_col, window_size, window_step, label_at=label_at
        )
        if window_size is None or window_size >= len(df_sorted):
            G = max(1, len(df_sorted))
        else:
            G = window_size
        groups = _groups_non_overlap_from_starts(starts, G=G)

        # 3) Split externo: escolhe 1 fold para TESTE
        sgkf_outer = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed_outer)
        outer_folds = list(sgkf_outer.split(np.zeros(len(y_win)), y=y_win, groups=groups))
        (trainval_idx, test_idx) = outer_folds[0]  # você pode iterar sobre vários folds se quiser

        # 4) Split interno (validação) sobre o restante
        sgkf_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed_inner)
        inner_folds = list(sgkf_inner.split(np.zeros(len(trainval_idx)),
                                            y=y_win[trainval_idx],
                                            groups=groups[trainval_idx]))
        (train_rel, val_rel) = inner_folds[0]
        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]

        # 5) DataLoaders com oversampling no TREINO
        #    (balanceia estados por janela)
        train_sampler = _make_weighted_sampler_from_classes(y_win[train_idx])
        train_loader = DataLoader(Subset(trainval_ds, train_idx),
                                batch_size=batch_size,
                                sampler=train_sampler,
                                pin_memory=True)

        val_loader = DataLoader(Subset(trainval_ds, val_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True) if validate else None

        test_loader = DataLoader(Subset(trainval_ds, test_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True)

        # 6) Loop de treino (referência: seu train_model)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
        self.to(device)

        best_score = float("inf")
        wait = patience

        for ep in range(1, epochs + 1):
            self.train()
            total_train = [[0.0, 0.0] for _ in range(4)]  # (valor acumulado, divisor) para L1 e L4
            for batch in train_loader:
                # batch = (x, ts, mask) ou (x, ts, mask, static)
                if len(batch) == 4:
                    x, ts_batch, m, s = batch
                else:
                    x, ts_batch, m = batch; s = None
                x = x.to(device, non_blocking=True)
                ts_batch = ts_batch.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                if s is not None: s = s.to(device, non_blocking=True)

                # máscara de treino (zero no último timestamp)
                m_train = m.clone()
                m_train[:, -1, :] = 0.0
                x_masked = x * m_train

                state, x_hat = self.forward(x_masked, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m_train)
                loss, L1, L4 = self._compute_loss(x, x_hat, state, m, m_train)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                for i, item in enumerate([L1, L4]):
                    total_train[i][0] += item[0]
                    total_train[i][1] += item[1]

            train_L1 = total_train[0][0] / max(total_train[0][1], 1.0)
            train_L4 = total_train[1][0] / max(total_train[1][1], 1.0)

            # --- Validação (macro-MSE por estado) ---
            if validate:
                val_metrics = self.test_model(val_loader, states_col=states_col, feature_cols=feature_cols)
                val_macro_mse = val_metrics["macro_mse"]
                print(f"Epoch {ep}/{epochs} | Train L1:{train_L1:.6f} L4:{train_L4:.6f} | Val macro-MSE:{val_macro_mse:.6f}")

                improved = val_macro_mse < best_score
                current_score = val_macro_mse
            else:
                # Se não validar, monitoro pelo teste (cuidado: uso só p/ early stopping)
                test_metrics = self.test_model(test_loader, states_col=states_col, feature_cols=feature_cols)
                test_macro_mse = test_metrics["macro_mse"]
                print(f"Epoch {ep}/{epochs} | Train L1:{train_L1:.6f} L4:{train_L4:.6f} | Test macro-MSE:{test_macro_mse:.6f}")
                improved = test_macro_mse < best_score
                current_score = test_macro_mse
                yield test_metrics  # pode ser útil para monitorar evolução no teste

            if early_stopping:
                if improved:
                    self.save("ode_jump.pt")
                    best_score = current_score
                    wait = patience
                else:
                    wait -= 1
                    if wait <= 0:
                        print(f"Early stopping at epoch {ep}/{epochs} (best macro-MSE: {best_score:.6f})")
                        break

        # --- Resultado final no TESTE ---
        final_metrics = self.test_model(test_loader, states_col=states_col, feature_cols=feature_cols)
        print(
            "TEST RESULTS | "
            f"macro-MSE: {final_metrics['macro_mse']:.6f} | micro-MSE: {final_metrics['micro_mse']:.6f} | "
            f"per_class: {final_metrics['per_class_mse']}"
        )
        yield None

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: list,
        static_features_cols: list,
        predict_state_cols: list,
        timestamp_col: str,
        status_pred_window: np.float32,
        batch_size: int = 32,
        lr: float = 1e-3,
        window_size: int = None,
        device: torch.device = None

    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_ds = self._make_dataset(df_train, timestamp_col, window_size, feature_cols, static_features_cols,predict_state_cols)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        if df_val is not None:
            val_ds = self._make_dataset(df_val, timestamp_col, window_size, feature_cols, static_features_cols, predict_state_cols)
            val_loader = DataLoader(val_ds, batch_size=batch_size) if df_val is not None else None
        
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.999),
                                      weight_decay=1e-2)
        total_train = [[0.0, 0.0] for _ in range(4)]
        self.to(device)
        self.train()
        for batch in train_loader:
            if len(batch) == 5:
                x, ts_batch, m, p, s = batch
            else:                               # caso não haja static
                x, ts_batch, m, p = batch;  s = None
            x, ts_batch, m, p = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True), p.to(device, non_blocking = True)
            if s is not None: s = s.to(device, non_blocking = True)
            m_train = m.copy()
            m_train[:, -1, :] = 0.0
            x_masked = x * m_train
            state, x_hat  = self.forward(
                x_masked, timestamps=ts_batch, static_feats=s, 
                return_x_hat=True, mask=m_train
                )
            # ---------- cabeças ----------
            loss,L1,L4= self._compute_loss(
                x, x_hat, state, m, m_train
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            for i,item in enumerate([L1,L4]):
                total_train[i][0]+=item[0]
                total_train[i][1]+=item[1]
        


        if df_val is not None:
            total_val = [[0.0, 0.0] for _ in range(4)]
            self.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 5:
                        x, ts_batch, m, p, s = batch
                    else:                               # caso não haja static
                        x, ts_batch, m, p = batch;  s = None
                    x, ts_batch, m, p = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True),p.to(device, non_blocking = True)
                    if s is not None: s = s.to(device, non_blocking = True)
                    t = torch.randint(0, self.num_steps, (x.size(0),), device=device)

                    # 2) probabilidade de *extra-missing* cresce com t
                    p_drop_t = (t.float() / (self.num_steps - 1)) * max_drop   # (B,)
                    p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
                    rand_mask = (torch.rand_like(m) > p_drop_t).float()
                    m_val   = m * rand_mask
                    x_masked = x * m_val
                    state , noise, x_hat, tmax = self.forward(x_masked, t, timestamps=ts_batch, static_feats=s, mask=m_val, return_x_hat=True, return_pred_state=True)
                    loss, L1, L2, L3, L4 = self._compute_loss(
                        x, x_hat, tmax,state, ts_batch, m, mask_train=m_val, noise=noise, state_pred=p, status_pred_window=status_pred_window
                    )
                    
                    for i,item in enumerate([L1,L2,L3,L4]):
                        total_val[i][0]+=item[0]
                        total_val[i][1]+=item[1]

            return total_train,total_val
        else:
            return total_train,None


    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model





if __name__ == '__main__':
    pass
            
