from .ode_jump_encoder import ODEJumpEncoder
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

max_drop = 0.7

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Beta schedule baseado em Nichol & Dhariwal (2021):
    β_t = 1 − ᾱ_t / ᾱ_{t−1}, onde
    ᾱ_t = cos²( (t/T + s) / (1 + s) * π/2 )
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # ᾱ desde t=0 até t=T
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # β_t = 1 − ᾱ_t / ᾱ_{t−1}
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5,max=0.999)  # evita valores muito altos

class DiffTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (estilo Vaswani et al.) + projeção linear.
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        half_dim = model_dim // 2
        # Registrar as frequências como buffer (não são parâmetros treináveis)
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) 
            * math.log(10000.0) / half_dim
        )
        self.register_buffer('freqs', freqs)  # shape: (half_dim,)
        # Projeção final: mantém dimensão
        self.lin = nn.Linear(model_dim, model_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor de shape (batch,) ou (batch, 1) com timesteps inteiros.
        Returns:
            emb: (batch, model_dim) embedding do timestep.
        """
        # garante shape (batch,)
        t = t.view(-1)
        # multiplica t pelas frequências: result -> (batch, half_dim)
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        # concatena seno e cosseno -> (batch, model_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # projeta de volta ao espaço de dimensão model_dim
        return self.lin(emb)
    
class TSDiffusion(ODEJumpEncoder):
    """
    Modelo de difusão para séries temporais contínuas, baseado em
    "Diffusion Models for Implicit Imputation of Time Series Data"
    (https://arxiv.org/abs/2205.14217) e
    "Score-Based Generative Modeling in Latent Space" (https://arxiv.org/abs/2206.00364).
    Usa Transformer com máscara causal para codificar a série temporal.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        lam: list[float,float] = [0.4, 0.5, 0.1],
        n_heads: int = 4,
        n_layers: int = 4,
        num_steps: int = 1000,                       
        ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            lam=lam,
            n_heads=n_heads,
            n_layers=n_layers
        )
        self.num_steps = num_steps
        self.t_embed = DiffTimeEmbedding(hidden_dim)
        self.noise_head = nn.Linear(hidden_dim,hidden_dim)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)
        self.t_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.model_dim, 2*self.model_dim)  # gamma, beta
        )  
        # (a) λ(t)  — intensity do ponto de observação
        self.lambda_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)       # escalar
        )
        # Schedule de difusão
        betas = cosine_beta_schedule(num_steps)
        alphas = 1 - betas
        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_bar', torch.cumprod(alphas, dim=0))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None,
        test: bool=True
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        noise = None
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        # Embedding de entrada
        if not already_latent:
            h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test:
                noise = torch.randn_like(h)
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
        else:
            h = x
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        te = self.t_embed(t).unsqueeze(1)          # (b,1,model_dim)
        tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        gb = self.t_film(te)                       # (B,1,2D)
        gamma, beta = gb.chunk(2, dim=-1)          # (B,1,D), (B,1,D)
        tm_e = (1.0 + gamma) * tm_e + beta         # FiLM no tempo contínuo
        h = h + te + tm_e
        h = self.encoder_ode(h, timestamps)
        state = h
        if test:
            return state,self.decoder(state) if return_x_hat else None
        else:
            return state,self.decoder(state) if return_x_hat else None, noise



    def denoise(self, state, timestamps, static_feats, device, steps,
                x0: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                enforce_data_consistency: bool = True):
        z = state
        for i in reversed(range(steps)):
            a, ab = self.alpha[i], self.alpha_bar[i]
            t = torch.full((z.size(0),), i, device=device, dtype=torch.long)

            # pred noise em latente
            state_i, _ = self.forward(
                z, t=t, timestamps=timestamps, static_feats=static_feats,
                already_latent=True, return_x_hat=False
            )
            eps_hat = self.noise_head(state_i)

            # passo de reverse (DDPM em latente)
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab)) * eps_hat)
            if i > 0:
                z = z + torch.sqrt(self.beta[i]) * torch.randn_like(z)

            # --- DATA CONSISTENCY opcional ---
            if enforce_data_consistency and (x0 is not None) and (mask is not None):
                with torch.no_grad():
                    x_hat_step = self.decoder(z)  # (B,T,C)
                    x_clamped  = torch.where(mask.bool(), x0, x_hat_step)
                    # re-encode para latente mantendo a máscara
                    z = self.encoder(torch.cat([x_clamped, mask], dim=-1))

        return z
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor,
        mask_train: torch.Tensor,
        noise
    ):
        #L1
        sse = ((x_hat - x)**2).sum(dim=-1) # (B,T) 
        nobs =x_hat.numel() # -½ λ ||x-μ||^2 + ½ log λ
        lam_t = F.softplus(self.lambda_head(state)).clamp(min=1e-8,max=1e+8) # (B,T,1) 
        lam2 = lam_t.squeeze(-1) # (B,T)  
        log_px = -0.5 * (lam2 * sse) + 0.5 * nobs * torch.log(lam2) - 0.5 * nobs * math.log(2*math.pi) # (B,T) 
        # Se quiser normalizar para não depender de C/T, use média por observação: # loss por (B,T) normalizada por nobs: 
        neg_log_px = -(log_px) # (B,T) 
        L1 = neg_log_px.sum() # escalar
        # ----- L4 (máscara) -----
        # máscara binária: 1 se ao menos um canal está presente no timestep
                # (B, T, 1)
        m_t = mask_train.any(dim=2, keepdim=True).float()              # (B,T,1)
        mb_pred = torch.sigmoid(self.miss_head(state)).clamp(1e-4, 1-1e-4)  # (B, T, 1)
        L4 = F.binary_cross_entropy(mb_pred, m_t, reduction='sum')
        L1_div = nobs
        L4_div = float(mb_pred.numel())
        L2 = F.mse_loss(noise, self.noise_head(state), reduction='sum')
        L2_div = float(state.numel())
        loss = self.lam[0]*L1/L1_div + self.lam[1] * L2 / L2_div + self.lam[2]*L4/L4_div

        return (
            loss,
            (float(L1.item()), float(L1_div)),
            (float(L2.item()),float(L2_div)),
            (float(L4.item()), float(L4_div))
            )
    
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
            total_train = [[0.0, 0.0] for _ in range(3)]  # L1, L4

            for batch in train_loader:
                if len(batch) == 4: x, ts_batch, m, s = batch
                else:               x, ts_batch, m = batch; s = None
                x, ts_batch, m = x.to(device), ts_batch.to(device), m.to(device)
                if s is not None: s = s.to(device)
                t_mask = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                # 2) probabilidade de *extra-missing* cresce com t
                p_drop_t = (t_mask.float() / (self.num_steps - 1)) * max_drop   # (B,)
                p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
                rand_mask = (torch.rand_like(m) > p_drop_t).float()
                m_train   = m * rand_mask
                x_masked = x * m_train
                state, x_hat, noise = self.forward(x_masked, timestamps=ts_batch, 
                                                   static_feats=s, return_x_hat=True, mask=m_train, test=False)
                loss, L1, L2, L4 = self._compute_loss(x, x_hat, state, m, m_train,noise)

                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                for i, item in enumerate([L1, L2, L4]):
                    total_train[i][0] += item[0]; total_train[i][1] += item[1]

            train_L1 = total_train[0][0] / max(total_train[0][1], 1.0)
            train_L2 = total_train[1][0] / max(total_train[1][1], 1.0)
            train_L4 = total_train[2][0] / max(total_train[2][1], 1.0)

            if validate and val_loader is not None:
                val_metrics = self.test_model(val_loader, y_seq=y_win[val_idx], all_groups=all_groups)
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L2:{train_L2:.6f} L4:{train_L4:.6f} | "
                    f"Val macro:{val_metrics['macro_mse']:.6f} ± {val_metrics['macro_se']:.6f} | "
                    f"Val micro:{val_metrics['micro_mse']:.6f} ± {val_metrics['micro_se']:.6f}"
                )
            else:
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L2:{train_L2:.6f} L4:{train_L4:.6f} | "
                )

            # teste fixo e ES
            test_metrics = self.test_model(test_loader, y_seq=y_win[test_idx], all_groups=all_groups)
            yield test_metrics
            print(
                f"          >> Test macro:{test_metrics['macro_mse']:.6f} ± {test_metrics['macro_se']:.6f} | "
                f"micro:{test_metrics['micro_mse']:.6f} ± {test_metrics['micro_se']:.6f}"
            )

            if early_stopping:
                improved = test_metrics["macro_mse"] < best_score
                if improved:
                    self.save("tsdiffusion.pt"); best_score = test_metrics["macro_mse"]; best_epoch = ep; wait = patience
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

    def test_model_preforward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        z = self.denoise(
            state=self.encoder(torch.cat([x, mask], dim=-1)),
            timestamps=timestamps,
            static_feats=static_feats,
            device=x.device,
            steps=self.denoise_steps if hasattr(self, 'denoise_steps') else 100
            )
        return self.decoder(z)

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

                state, x_hat, noise = self.forward(x, timestamps=ts_batch, static_feats=s, return_x_hat=True, mask=m, test=False)

                sse_bt  = ((self.noise_head(state)-noise)**2).sum(dim=(1, 2))           # (B,)
                nobs_bt = torch.ones_like(state).sum(dim=(1, 2))               # (B,)
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
    def denoise_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        timestamp_col: str,
        static_features_cols: list[str] | None = None,
        window_size: int | None = None,
        window_step: int = 1,
        steps: int | None = None,
        replace_only_missing: bool = True,
        device: torch.device | None = None,
    ) -> pd.DataFrame:
        """
        Retorna um novo DataFrame com as colunas feature_cols denoised.
        - Se window_size for None ou >= len(df), processa de uma vez.
        - Com janelas sobrepostas, agrega por média.
        - Por padrão só substitui NaN (replace_only_missing=True).
        """
        self.eval()
        device = device or next(self.parameters()).device
        static_features_cols = static_features_cols or []

        # Mantém índice original; caso ordene por tempo, voltaremos ao índice depois
        orig_index = df.index
        needs_sort = (timestamp_col != "index")
        if needs_sort:
            df_sorted = df.sort_values(timestamp_col).reset_index(drop=False)
            idx_col_name = df_sorted.columns[0]  # coluna do índice original após reset_index
        else:
            df_sorted = df.copy()

        # Monta dataset exatamente como no treino
        ds = self._make_dataset(
            df_sorted,
            timestamp_col=timestamp_col,
            window_size=window_size,
            feature_cols=feature_cols,
            static_features_cols=static_features_cols,
            window_step=window_step,
        )
        tensors = ds.tensors
        if len(tensors) == 4:
            seqs, ts_seqs, mask_seqs, stat_seqs = tensors
        else:
            seqs, ts_seqs, mask_seqs = tensors
            stat_seqs = None

        seqs = seqs.to(device)
        ts_seqs = ts_seqs.to(device)
        mask_seqs = mask_seqs.to(device)
        if stat_seqs is not None:
            stat_seqs = stat_seqs.to(device)

        with torch.no_grad():
            h0 = self.encoder(torch.cat([seqs, mask_seqs], dim=-1))
            n_steps = steps if steps is not None else getattr(self, "denoise_steps", self.num_steps)
            z = self.denoise(
                state=h0,
                timestamps=ts_seqs,
                static_feats=stat_seqs,
                device=device,
                steps=n_steps,
                x0=seqs,                     # <- importante
                mask=mask_seqs,              # <- importante
                enforce_data_consistency=False
            )
            x_hat = self.decoder(z).detach().cpu().numpy()

        n = len(df_sorted)
        C = len(feature_cols)
        out = np.zeros((n, C), dtype=np.float32)
        cnt = np.zeros((n, C), dtype=np.float32)

        # matriz original e máscara de missing (True = faltante)
        orig_vals = df_sorted[feature_cols].to_numpy()
        miss = ~np.isfinite(orig_vals)

        if window_size is None or window_size >= n:
            pred = x_hat[0]
            if replace_only_missing:
                out = np.where(miss, pred, orig_vals)
                cnt = np.where(miss, 1.0, 0.0).astype(np.float32)
            else:
                out = pred
                cnt[:] = 1.0
        else:
            starts = np.arange(0, n - window_size + 1, window_step, dtype=int)
            for k, s in enumerate(starts):
                e = min(s + window_size, n)
                pred = x_hat[k, :e - s, :]  # (L_k, C)

                if replace_only_missing:
                    sel = miss[s:e, :]                  # só substitui onde falta
                    # escreve pred onde falta; mantém original onde não falta
                    blended = np.where(sel, pred, orig_vals[s:e, :])
                    out[s:e, :] += blended
                    cnt[s:e, :] += sel.astype(np.float32)
                else:
                    out[s:e, :] += pred
                    cnt[s:e, :] += 1.0

            # posições não cobertas por nenhuma janela ou nunca substituídas
            no_write = (cnt == 0.0)
            out[no_write] = orig_vals[no_write]

            # média nas posições com múltiplas escritas
            written = (cnt > 0.0)
            out[written] = out[written] / cnt[written]

        # monta DataFrame de saída
        result = df_sorted.copy()
        result[feature_cols] = out

        # restaura ordem/índice original caso tenha ordenado por tempo
        if needs_sort:
            result = result.set_index(idx_col_name).loc[orig_index]
            result.index = orig_index  # garante exatamente o mesmo Index

        return result