import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np 
max_drop = 0.7
TS_SPAN = 60 * 60 * 24 * 365
from sklearn.model_selection import StratifiedShuffleSplit  # manter se já importou, usamos só p/ seed-size; NÃO estratifica tempo

# ---------- Helpers privados ----------

def _split_counts(n, train_frac, val_frac, test_frac, validate):
    # contas base (arredonda e corrige sobra/falta)
    if validate:
        raw = np.array([train_frac, val_frac, test_frac], dtype=float)
    else:
        raw = np.array([train_frac + val_frac, 0.0, test_frac], dtype=float)  # 80/0/20

    k = raw / max(raw.sum(), 1e-8)
    counts = np.floor(k * n).astype(int)
    # distribui sobras para atingir n
    while counts.sum() < n:
        # dá 1 para o maior "gap" relativo (quase sempre treino)
        gaps = k * n - counts
        j = int(np.argmax(gaps))
        counts[j] += 1
    # se passou (por arredondamento), tira de quem tem mais
    while counts.sum() > n:
        j = int(np.argmax(counts))
        counts[j] -= 1
    # garante não-negativos
    counts = np.maximum(counts, 0)
    if validate:
        n_train, n_val, n_test = counts.tolist()
        # tenta garantir ao menos 1 em val/test se couber
        if n >= 3:
            if n_val == 0: n_val, n_train = 1, max(n_train-1, 0)
            if n_test == 0:
                if n_train > n_val: n_test, n_train = 1, max(n_train-1, 0)
                else:               n_test, n_val   = 1, max(n_val-1,   0)
    else:
        n_train, n_val, n_test = counts.tolist()  # n_val==0
    # ajuste final se estourar
    extra = n_train + n_val + n_test - n
    if extra > 0:
        for _ in range(extra):
            # tira de onde tem mais (preferindo treino)
            if n_train >= max(n_val, n_test) and n_train > 0:
                n_train -= 1
            elif n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
    return n_train, n_val, n_test








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

    @staticmethod
    def _make_weighted_sampler_from_classes(y_classes: np.ndarray):
        """Peso inverso à frequência da classe (estado)."""
        binc = np.bincount(y_classes)
        binc = np.maximum(binc, 1)
        w_per_class = 1.0 / binc
        weights = w_per_class[y_classes]
        weights = torch.as_tensor(weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),   # um 'epoch' lógico
            replacement=True
        )
        return sampler

    @staticmethod
    def _split_by_group_proportions(y_win: np.ndarray,
                                    validate: bool,
                                    train_frac: float = 0.60,
                                    val_frac: float = 0.20,
                                    test_frac: float = 0.20,
                                    seed: int = 42):
        """
        Split simples por grupo (state):
        - embaralha índices DENTRO de cada grupo com seed.
        - reparte por proporções.
        - concatena (sem ordenação temporal).
        """
        rng = np.random.default_rng(seed)
        idx_all = np.arange(len(y_win), dtype=int)
        train_idx, val_idx, test_idx = [], [], []

        groups = np.unique(y_win)
        for g in groups:
            g_idx = idx_all[y_win == g].copy()
            rng.shuffle(g_idx)
            n = len(g_idx)
            n_train, n_val, n_test = _split_counts(n, train_frac, val_frac, test_frac, validate)

            # fatiamento: test pega do fim para reduzir correlação com treino
            # (tanto faz aqui, pois não olhamos tempo; apenas mantemos aleatório)
            test_idx.extend(g_idx[:n_test])
            val_idx.extend(g_idx[n_test:n_test+n_val])
            train_idx.extend(g_idx[n_test+n_val:])

        # Embaralha a ordem final (para DataLoader iterar misto)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)    
    @staticmethod
    def _states_from_df_windows(
        df: pd.DataFrame,
        states_col,                 # str (rótulo) OU List[str] (one-hot/prob, >=2 colunas)
        window_size: int,
        window_step: int,
        label_at: str = "end",
    ):
        n_rows = len(df)
        if window_size is None or window_size >= n_rows:
            starts = np.array([0], dtype=int)
        else:
            starts = np.arange(0, n_rows - window_size + 1, window_step, dtype=int)

        # rótulo por timestamp (y_ts: (L,))
        if isinstance(states_col, str):
            col = df[states_col]
            if not np.issubdtype(col.dtype, np.number):
                y_ts = pd.Categorical(col).codes.astype(int)
            else:
                y_vals = col.to_numpy()
                if np.issubdtype(y_vals.dtype, np.floating) and not np.allclose(y_vals, np.round(y_vals)):
                    y_vals = np.round(y_vals)
                y_ts = y_vals.astype(int)
        else:
            if len(states_col) < 2:
                raise ValueError("Se states_col for lista, deve ter >=2 colunas (one-hot/prob). Para coluna única, passe str.")
            S = df[states_col].to_numpy(dtype=np.float32)  # (L, S)
            y_ts = np.nanargmax(S, axis=1).astype(int)

        # rótulo por janela
        y_win = np.empty(len(starts), dtype=int)
        for k, s in enumerate(starts):
            e = min(s + window_size, n_rows)
            if label_at == "end":
                y_win[k] = y_ts[e - 1]
            else:
                seg = y_ts[s:e]
                vals, cnts = np.unique(seg, return_counts=True)
                y_win[k] = vals[np.argmax(cnts)]

        return y_win, starts

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
    
    def test_model_preforward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        return x

        # ---------- Novo método: test_model (macro/micro e por estado) ----------
    def test_model(self, loader: DataLoader, y_seq, all_groups=None):
        """
        Avalia reconstrução por janela e retorna:
        - micro_mse, micro_se            (ponderado por nobs, todas as janelas)
        - macro_mse, macro_se            (média das MÉDIAS por grupo; SE entre grupos, não-ponderado)
        - per_group_mse                  (média ponderada por nobs dentro do grupo)
        - per_group_se_w                 (SE ponderado por nobs dentro do grupo; usa n_eff)
        - per_group_se_unw               (SE não-ponderado dentro do grupo; diagnóstico)
        - per_group_counts, per_group_sum_nobs
        Fórmulas de SE:
        - Não-ponderado: SE = std_amostral / sqrt(n)
        - Ponderado:     SE = sqrt( s2_w / n_eff ), onde
                            s2_w = Σ α_i (x_i - μ)^2  e  n_eff = 1 / Σ α_i^2, α_i = w_i / Σ w_i
        """
        import math
        device = next(self.parameters()).device
        self.eval()

        y_seq = np.asarray(y_seq, dtype=int)
        pos = 0

        # acumuladores por grupo
        G_W      = {}   # sum w = sum nobs
        G_SSE    = {}   # sum sse = sum w * mse
        G_WM2    = {}   # sum w * mse^2
        G_MSE    = {}   # lista de mse por janela (p/ SE não-ponderado)
        G_CNT    = {}   # nº janelas

        # globais (micro)
        T_W, T_SSE, T_WM2 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x, ts_batch, m, s = batch
                else:
                    x, ts_batch, m = batch; s = None
                x, ts_batch, m = x.to(device), ts_batch.to(device), m.to(device)
                if s is not None: s = s.to(device)

                B = x.shape[0]
                yb = y_seq[pos:pos+B]
                if len(yb) != B:
                    raise ValueError(f"test_model: desalinhado (batch={B}, labels={len(yb)} a partir de pos={pos}).")
                pos += B

                m_train = m.clone(); m_train[:, -1, :] = 0.0
                x_masked = x * m_train
                x_masked = self.test_model_preforward(x_masked, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m_train)
                state, x_hat = self.forward(x_masked, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m_train)

                mask_err = m * (1.0 - m_train)
                sse_bt  = ((x - x_hat)**2 * mask_err).sum(dim=(1, 2))               # (B,)
                nobs_bt = mask_err.sum(dim=(1, 2)).clamp(min=1.0)                   # (B,)
                mse_bt  = (sse_bt / nobs_bt).detach().cpu().numpy()
                sse_bt  = sse_bt.detach().cpu().numpy()
                nobs_bt = nobs_bt.detach().cpu().numpy()

                for b in range(B):
                    g   = int(yb[b])
                    w   = float(nobs_bt[b])
                    mse = float(mse_bt[b])
                    sse = float(sse_bt[b])

                    G_W[g]   = G_W.get(g, 0.0)   + w
                    G_SSE[g] = G_SSE.get(g, 0.0) + sse
                    G_WM2[g] = G_WM2.get(g, 0.0) + (w * mse * mse)
                    G_MSE.setdefault(g, []).append(mse)
                    G_CNT[g] = G_CNT.get(g, 0) + 1

                    T_W   += w
                    T_SSE += sse
                    T_WM2 += (w * mse * mse)

        # grupos a reportar
        if all_groups is None:
            groups = sorted(G_W.keys())
        else:
            groups = sorted(np.unique(list(all_groups)).tolist())

        if not groups:
            return {
                "macro_mse": float("nan"), "macro_se": float("nan"),
                "micro_mse": float("nan"), "micro_se": float("nan"),
                "per_group_mse": {}, "per_group_se_w": {}, "per_group_se_unw": {},
                "per_group_counts": {}, "per_group_sum_nobs": {}
            }

        # por grupo
        per_group_mse       = {}
        per_group_se_w      = {}
        per_group_se_unw    = {}
        per_group_counts    = {}
        per_group_sum_nobs  = {}

        for g in groups:
            Wg = G_W.get(g, 0.0)
            per_group_sum_nobs[g] = float(Wg)
            cnt = G_CNT.get(g, 0)
            per_group_counts[g] = int(cnt)

            if Wg > 0.0:
                mu_g = G_SSE[g] / Wg                           # média ponderada por nobs
                per_group_mse[g] = float(mu_g)

                # SE não-ponderado (amostral) entre janelas
                mses = np.asarray(G_MSE.get(g, []), dtype=float)
                if mses.size >= 2:
                    std_unw = float(np.std(mses, ddof=1))
                    per_group_se_unw[g] = std_unw / math.sqrt(mses.size)
                elif mses.size == 1:
                    per_group_se_unw[g] = float("nan")
                else:
                    per_group_se_unw[g] = float("nan")

                # SE ponderado por nobs (usa n_eff)
                # s2_w = E_w[(X - mu)^2] = (Σ w x^2)/Wg - mu_g^2
                s2_w = max(G_WM2[g] / Wg - mu_g * mu_g, 0.0)
                # n_eff = 1 / Σ α_i^2, com α_i = w_i / Wg
                # para computar Σ α_i^2, precisamos das α_i por janela do grupo:
                # reusa mses + w por grupo (não armazenamos w_i individuais por grupo; então recompute α_i via segunda passada)
                # -> atalho: acumule Σ w_i^2 enquanto itera (opção mais eficiente).
                # Como não acumulamos, aproximamos n_eff por contagem não-ponderada quando não há forte desbalanceamento:
                # Melhor: compute n_eff aproximado por (Wg^2) / Σ w_i^2 – para isso, reconstruímos Σ w_i^2 do G_WM2 e s2_w:
                # G_WM2 = Σ w_i x_i^2. Não temos Σ w_i^2 diretamente; então usamos aproximação conservadora n_eff = cnt.
                # Se quiser exato, guarde Σ w_i^2 durante a passada no loader.
                if cnt >= 2:
                    n_eff = cnt  # aproximação segura; se quiser exato, armazene sum_w2 por grupo
                    per_group_se_w[g] = float(math.sqrt(s2_w / n_eff))
                else:
                    per_group_se_w[g] = float("nan")
            else:
                per_group_mse[g]    = float("nan")
                per_group_se_unw[g] = float("nan")
                per_group_se_w[g]   = float("nan")

        # micro (ponderado por nobs) – SE ponderado
        if T_W > 0.0:
            micro_mse = T_SSE / T_W
            # var ponderada populacional
            s2_micro = max(T_WM2 / T_W - micro_mse * micro_mse, 0.0)
            # n_eff global (aprox): use número de janelas (contagem total) como proxy
            # Para n_eff exato, acumule Σ w_i^2 globalmente. Se puder, acrescente 'sum_w2' no loop.
            total_cnt = int(sum(per_group_counts.values()))
            micro_se = float(math.sqrt(s2_micro / max(total_cnt, 1)))
        else:
            micro_mse = float("nan"); micro_se = float("nan")

        # macro: média das MÉDIAS por grupo (não-ponderado) e SE entre grupos
        mu_gs = [per_group_mse[g] for g in groups if np.isfinite(per_group_mse[g])]
        G_eff = len(mu_gs)
        if G_eff >= 1:
            macro_mse = float(np.mean(mu_gs))
            if G_eff >= 2:
                std_between = float(np.std(mu_gs, ddof=1))
                macro_se = std_between / math.sqrt(G_eff)
            else:
                macro_se = float("nan")
        else:
            macro_mse = float("nan"); macro_se = float("nan")

        return {
            "macro_mse": macro_mse, "macro_se": macro_se,
            "micro_mse": micro_mse, "micro_se": micro_se,
            "per_group_mse": per_group_mse,
            "per_group_se_w": per_group_se_w,
            "per_group_se_unw": per_group_se_unw,
            "per_group_counts": per_group_counts,
            "per_group_sum_nobs": per_group_sum_nobs
        }

    
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
            starts = np.arange(0, len(df) - window_size + 1, window_step, dtype=int)
            seqs      = torch.stack([data[s:s+window_size]  for s in starts])
            ts_seqs   = torch.stack([times[s:s+window_size] for s in starts])
            mask_seqs = torch.stack([mask[s:s+window_size]  for s in starts])
            stat_seqs = static[0].unsqueeze(0).repeat(len(starts), 1) if static is not None else None
        if stat_seqs is None:
            return TensorDataset(seqs, ts_seqs, mask_seqs)                   # 3 itens
        return TensorDataset(seqs, ts_seqs, mask_seqs, stat_seqs)   

    # ---------- Novo método: train_cognite ----------
    def train_cognite(self,
        df: pd.DataFrame,
        feature_cols: list,
        static_features_cols: list,
        timestamp_col: str,
        states_col: str | list,
        batch_size: int = 32,
        lr: float = 3e-4,
        window_size: int = None,
        window_step: int = 1,
        epochs: int = 10,
        validate: bool = True,
        early_stopping: bool = True,
        patience: int = 5,
        device: torch.device = None,
        label_at: str = "end",
        fixed_test_idx: np.ndarray | None = None,
        seed_split: int = 42,
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_sorted = df if timestamp_col == "index" else df.sort_values(timestamp_col).reset_index(drop=True)

        # Dataset (sem y) e rótulos de grupo por janela (para split/oversampling/relato)
        ds = self._make_dataset(df_sorted, timestamp_col, window_size, feature_cols, static_features_cols, window_step)
        y_win, _starts = self._states_from_df_windows(df_sorted, states_col, window_size, window_step, label_at)
        all_groups = np.unique(y_win)

        N = ds.tensors[0].shape[0]
        if N != len(y_win):
            raise ValueError(f"Inconsistência: dataset={N} vs rótulos={len(y_win)}.")

        # --- Split por grupo (proporcional): 60/20/20 ou 80/0/20
        if fixed_test_idx is not None:
            test_idx = np.asarray(fixed_test_idx, dtype=int)
            remain_mask = np.ones(N, dtype=bool); remain_mask[test_idx] = False
            tr_idx_rel, va_idx_rel, _ = self._split_by_group_proportions(
                y_win[remain_mask], validate=validate,
                train_frac=0.60, val_frac=0.20, test_frac=0.20, seed=seed_split
            )
            base = np.where(remain_mask)[0]
            train_idx = base[tr_idx_rel]
            val_idx   = base[va_idx_rel]
        else:
            train_idx, val_idx, test_idx = self._split_by_group_proportions(
                y_win, validate=validate, train_frac=0.60, val_frac=0.20, test_frac=0.20, seed=seed_split
            )

        # --- Oversampling APENAS no treino
        train_sampler = self._make_weighted_sampler_from_classes(y_win[train_idx]) if len(train_idx) else None

        # --- DataLoaders
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                                sampler=train_sampler if train_sampler is not None else None,
                                shuffle=False, pin_memory=True)
        # avaliador do treino na distribuição real (sem oversampling)
        val_loader  = DataLoader(Subset(ds, val_idx),  batch_size=batch_size,
                                shuffle=False, pin_memory=True) if validate and len(val_idx) else None
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size,
                                shuffle=False, pin_memory=True)

        # --- Logs de cobertura de grupos
        def _count(y):
            keys, cnts = np.unique(y, return_counts=True)
            return dict(zip(keys.tolist(), cnts.tolist()))
        print("GRUPOS (total):", _count(y_win))
        print("GRUPOS (train):", _count(y_win[train_idx]))
        if validate and len(val_idx): print("GRUPOS (valid):", _count(y_win[val_idx]))
        print("GRUPOS (test): ", _count(y_win[test_idx]))

        # --- Treino + ES sempre no TESTE
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
        self.to(device)
        best_score = float("inf"); best_epoch = 0; wait = patience

        for ep in range(1, epochs + 1):
            self.train()
            total_train = [[0.0, 0.0] for _ in range(2)]  # L1, L4

            for batch in train_loader:
                if len(batch) == 4: x, ts_batch, m, s = batch
                else:               x, ts_batch, m = batch; s = None
                x, ts_batch, m = x.to(device), ts_batch.to(device), m.to(device)
                if s is not None: s = s.to(device)

                m_train = m.clone(); m_train[:, -1, :] = 0.0
                x_masked = x * m_train
                state, x_hat = self.forward(x_masked, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m_train)
                loss, L1, L4 = self._compute_loss(x, x_hat, state, m, m_train)

                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                for i, item in enumerate([L1, L4]):
                    total_train[i][0] += item[0]; total_train[i][1] += item[1]

            train_L1 = total_train[0][0] / max(total_train[0][1], 1.0)
            train_L4 = total_train[1][0] / max(total_train[1][1], 1.0)

            if validate and val_loader is not None:
                val_metrics = self.test_model(val_loader, y_seq=y_win[val_idx], all_groups=all_groups)
                yield val_metrics
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L4:{train_L4:.6f} | "
                    f"Val macro:{val_metrics['macro_mse']:.6f} ± {val_metrics['macro_se']:.6f} | "
                    f"Val micro:{val_metrics['micro_mse']:.6f} ± {val_metrics['micro_se']:.6f}"
                )
            else:
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L4:{train_L4:.6f} | "
                )

            # teste fixo e ES
            test_metrics = self.test_model(test_loader, y_seq=y_win[test_idx], all_groups=all_groups)
            if not validate:
                yield test_metrics
            print(
                f"          >> Test macro:{test_metrics['macro_mse']:.6f} ± {test_metrics['macro_se']:.6f} | "
                f"micro:{test_metrics['micro_mse']:.6f} ± {test_metrics['micro_se']:.6f}"
            )

            if early_stopping:
                improved = test_metrics["macro_mse"] < best_score
                if improved:
                    self.save("ode_jump.pt"); best_score = test_metrics["macro_mse"]; best_epoch = ep; wait = patience
                else:
                    wait -= 1
                    if wait <= 0:
                        print(f"Early stopping at epoch {ep}/{epochs} (best test macro-MSE: {best_score:.6f} @ epoch {best_epoch})")
                        break

        # --- Resultado final no teste fixo
        final_metrics = self.test_model(test_loader, y_seq=y_win[test_idx], all_groups=all_groups)
        print(
            "TEST RESULTS | "
            f"macro: {final_metrics['macro_mse']:.6f} ± {final_metrics['macro_se']:.6f} | "
            f"micro: {final_metrics['micro_mse']:.6f} ± {final_metrics['micro_se']:.6f}"
        )
        pg = test_metrics["per_group_mse"]
        pg_sew = test_metrics["per_group_se_w"]
        pg_cnt = test_metrics["per_group_counts"]
        print("          >> per_group (weighted SE):",
        {g: f"{pg[g]:.6f} ± {pg_sew[g]:.6f} (n={pg_cnt[g]})" for g in sorted(pg.keys())}
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
            
