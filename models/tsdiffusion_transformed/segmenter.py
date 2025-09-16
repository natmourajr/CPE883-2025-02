# Update: Multivariate support (multiple channels) for UnsupervisedStateSegmenter.
# Changes:
# - Accept series shape (T,) or (T, C).
# - Windowing keeps channels: (num_windows, window, C).
# - Per-channel features computed and concatenated across channels.
# - Cross-channel features: pairwise Pearson corr (lag 0) and covariances per window.
# - ACF and bandpowers computed per channel and concatenated.
# - API unchanged (`fit`, `predict`).


from typing import Tuple, List, Dict, Optional, Iterable
import numpy as np
import torch
from dataclasses import dataclass
from scipy.signal import periodogram
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import itertools
import pickle
from copy import deepcopy


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    req = int(max_lag)                   # o que foi solicitado
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    if req <= 0:
        return np.zeros(0, dtype=float)

    # Se série é constante ou muito curta, devolve zeros com o TAMANHO SOLICITADO
    if n <= 1 or np.allclose(x.var(), 0.0):
        return np.zeros(req, dtype=float)

    eff = min(req, n - 1)               # lag efetivo suportado pela janela
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acf_full = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    acf_full /= acf_full[0] if acf_full[0] != 0 else 1.0

    out = np.real(acf_full[1:eff + 1])  # tamanho 'eff'
    # Agora pad para o TAMANHO SOLICITADO (req), não para 'eff'
    if len(out) < req:
        out = np.pad(out, (0, req - len(out)), constant_values=0.0)
    return out



def _ols_slope(y: np.ndarray) -> float:
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = (n - 1) / 2.0
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def _fft_bandpowers(y: np.ndarray, fs: float, bands: List[Tuple[float, float]]) -> np.ndarray:
    if len(y) < 2 or fs <= 0:
        return np.zeros(len(bands), dtype=float)
    f, pxx = periodogram(y, fs=fs, scaling="density")
    out = []
    for fmin, fmax in bands:
        mask = (f >= fmin) & (f < fmax)
        out.append(float(np.mean(pxx[mask])) if np.any(mask) else 0.0)
    return np.array(out, dtype=float)


def _percent_outliers_mad(y: np.ndarray, thresh: float = 3.0) -> float:
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    z = np.abs(y - med) / (1.4826 * mad)
    return float((z > thresh).mean())


@dataclass
class FeatureConfig:
    acf_lags: int = 10
    bands: Optional[List[Tuple[float, float]]] = None
    fs: float = 1.0


class UnsupervisedStateSegmenter:
    def __init__(
        self,
        k_states: int = 3,
        window: int = 50,
        step: int = 10,
        min_slice_windows: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        cov_type: str = "diag",
        anomaly_strategy: str = "lowest_ll",
        feature_conf: Optional[FeatureConfig] = None,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.k = k_states
        self.w = window
        self.step = step
        self.min_slice = min_slice_windows
        self.max_iter = max_iter
        self.tol = tol
        self.cov_type = cov_type
        self.anomaly_strategy = anomaly_strategy
        self.feature_conf = feature_conf or FeatureConfig(acf_lags=10, bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.5)], fs=1.0)
        self.device = torch.device(device)
        self.random_state = random_state

        self.scaler_: Optional[RobustScaler] = None
        self.mu_: Optional[torch.Tensor] = None
        self.var_: Optional[torch.Tensor] = None
        self.pi_: Optional[torch.Tensor] = None
        self.A_: Optional[torch.Tensor] = None
        self.loglik_: List[float] = []
        self.anomaly_state_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    # ---------------- NEW: Serialização em memória ----------------
    def to_state_dict(self) -> Dict:
        """Exporta todo o estado necessário para inferência."""
        if not self.is_trained():
            raise RuntimeError("Modelo não está treinado/carregado; execute fit() ou load() antes de salvar.")

        # Serializar scaler (classe + hiperparâmetros + estatísticas)
        scaler_state = {
            "class": "RobustScaler",
            "params": self.scaler_.get_params(),
            "center_": deepcopy(getattr(self.scaler_, "center_", None)),
            "scale_":  deepcopy(getattr(self.scaler_, "scale_",  None)),
            "n_features_in_": int(getattr(self.scaler_, "n_features_in_", len(self.feature_names_ or []))),
        }

        # Serializar FeatureConfig (usável para reconstrução)
        fc = self.feature_conf
        feature_conf_state = {
            "acf_lags": int(fc.acf_lags),
            "bands": deepcopy(fc.bands),
            "fs": float(fc.fs),
        }

        # Tensores -> numpy
        def t2n(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        state = {
            "init_params": {
                "k_states": self.k,
                "window": self.w,
                "step": self.step,
                "min_slice_windows": self.min_slice,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "cov_type": self.cov_type,
                "anomaly_strategy": self.anomaly_strategy,
                "feature_conf": feature_conf_state,
                "device": str(self.device),        # apenas para informação
                "random_state": self.random_state,
            },
            "scaler_state": scaler_state,
            "mu": t2n(self.mu_),
            "var": t2n(self.var_),
            "pi": t2n(self.pi_),
            "A":  t2n(self.A_),
            "loglik": deepcopy(self.loglik_),
            "anomaly_state": int(self.anomaly_state_),
            "feature_names": deepcopy(self.feature_names_),
        }
        return state

    @classmethod
    def from_state_dict(cls, state: Dict, map_location: str = "cpu") -> "UnsupervisedStateSegmenter":
        """Restaura uma instância completamente pronta para inferência a partir de um state dict."""
        ip = state["init_params"]
        # Reconstruir FeatureConfig
        fc_s = ip["feature_conf"]
        feature_conf = FeatureConfig(
            acf_lags=int(fc_s["acf_lags"]),
            bands=deepcopy(fc_s["bands"]),
            fs=float(fc_s["fs"]),
        )

        # Criar instância (device pode ser sobrescrito depois)
        inst = cls(
            k_states=int(ip["k_states"]),
            window=int(ip["window"]),
            step=int(ip["step"]),
            min_slice_windows=int(ip["min_slice_windows"]),
            max_iter=int(ip["max_iter"]),
            tol=float(ip["tol"]),
            cov_type=ip["cov_type"],
            anomaly_strategy=ip["anomaly_strategy"],
            feature_conf=feature_conf,
            device=map_location,                     # escolhemos o device de restauração
            random_state=int(ip["random_state"]),
        )

        # Restaurar scaler
        scs = state["scaler_state"]
        scaler = RobustScaler(**scs["params"])
        # Forçar atributos de fitted:
        scaler.center_ = np.asarray(scs["center_"]) if scs["center_"] is not None else None
        scaler.scale_  = np.asarray(scs["scale_"])  if scs["scale_"]  is not None else None
        # Ajustar n_features_in_ para evitar warnings/erros no transform
        scaler.n_features_in_ = int(scs.get("n_features_in_", len(state.get("feature_names", []) or [])))
        inst.scaler_ = scaler

        # Restaurar tensores no device alvo
        dev = torch.device(map_location)
        inst.mu_  = torch.as_tensor(state["mu"],  dtype=torch.float32, device=dev)
        inst.var_ = torch.as_tensor(state["var"], dtype=torch.float32, device=dev)
        inst.pi_  = torch.as_tensor(state["pi"],  dtype=torch.float32, device=dev)
        inst.A_   = torch.as_tensor(state["A"],   dtype=torch.float32, device=dev)

        inst.loglik_ = list(state.get("loglik", []))
        inst.anomaly_state_ = int(state["anomaly_state"])
        inst.feature_names_ = list(state["feature_names"])
        inst.device = dev
        return inst

    # ---------------- NEW: Persistência em disco ----------------
    def save(self, path: str) -> None:
        """Salva o estado pronto para inferência em disco (pickle)."""
        state = self.to_state_dict()
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "UnsupervisedStateSegmenter":
        """Carrega o estado salvo e retorna uma instância pronta para predict."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls.from_state_dict(state, map_location=map_location)

    def is_trained(self) -> bool:
        """Retorna True se o modelo está pronto para prever (fit ou load)."""
        return (
            self.scaler_ is not None and
            self.mu_ is not None and
            self.var_ is not None and
            self.pi_ is not None and
            self.A_ is not None and
            self.feature_names_ is not None and
            self.anomaly_state_ is not None
        )
    # ---------- Public API ----------

    def fit(self, series: np.ndarray) -> "UnsupervisedStateSegmenter":
        X, idx = self._build_windows(series)
        F = self._extract_features(X)          # (T, d)
        Fz = self._scale(F)
        self._init_params(Fz)
        self._em_train(Fz)
        z = self.viterbi(Fz)
        self._choose_anomaly_state(Fz, z)
        return self

    def predict(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        # --------- NEW: garantir que há estado carregado/treinado ---------
        if not self.is_trained():
            raise RuntimeError("Modelo não está pronto para inferência. Chame fit() ou load() antes de predict().")

        X, idx = self._build_windows(series)
        F = self._extract_features(X)
        Fz = self._scale(F, fit=False)
        z = self.viterbi(Fz)
        logB = self._log_emission(Fz)  # (T, k)
        ll = torch.logsumexp(logB + torch.log(self.pi_.unsqueeze(0)), dim=1).detach().cpu().numpy()

        anomaly_mask = (z == self.anomaly_state_).astype(int)
        slices = self._build_slices(z)
        slices = self._enforce_min_dwell(slices)
        states = []
        for i,v in enumerate(idx):
            states += [z[i]] * (v[1] - v[0])
        states = np.array(states, dtype=int)
        return {
            "window_indices": np.array(idx, dtype=int),
            "features": F,
            "features_z": Fz.cpu().numpy(),
            "states": states,
            "states_per_window": z,
            "loglike_per_window": ll,
            "anomaly_state": np.array([self.anomaly_state_]),
            "anomaly_mask": anomaly_mask,
            "slices": np.array(slices, dtype=int),
            "feature_names": np.array(self.feature_names_),
        }
    
    # ---------- Windowing & Features ----------

    def _build_windows(self, series: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        x = np.asarray(series, dtype=float)
        if x.ndim == 1:
            x = x[:, None]  # (T, 1)
        T, C = x.shape
        windows = []
        indices = []
        for start in range(0, max(1, T - self.w + 1), self.step):
            end = start + self.w
            if end <= T:
                windows.append(x[start:end, :])  # (w, C)
                indices.append((start, end))
        X = np.stack(windows, axis=0)  # (Nw, w, C)
        return X, indices

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        Nw, w, C = X.shape
        fc = self.feature_conf
        feats = []
        names_once = None

        # Precompute channel pairs for cross-channel features
        pairs = list(itertools.combinations(range(C), 2))

        for t in range(Nw):
            Y = X[t]  # (w, C)
            f_all = []

            # Per-channel features
            for c in range(C):
                y = Y[:, c]
                slope = _ols_slope(y)
                y_detr = y - (np.arange(w) * slope + y[0])

                mean = float(np.mean(y))
                median = float(np.median(y))
                std = float(np.std(y, ddof=1)) if w > 1 else 0.0
                iqr = float(np.subtract(*np.percentile(y, [75, 25])))
                rng = float(np.max(y) - np.min(y))
                skew = float(((y - mean) ** 3).mean() / (std ** 3 + 1e-12)) if std > 0 else 0.0
                kurt = float(((y - mean) ** 4).mean() / (std ** 4 + 1e-12)) if std > 0 else 0.0
                acf = _acf_1d(y, fc.acf_lags)
                mad_out = _percent_outliers_mad(y)
                band_powers = _fft_bandpowers(y_detr, fs=fc.fs, bands=fc.bands or [])

                f_ch = [mean, median, std, iqr, rng, skew, kurt, slope, mad_out]
                f_ch = np.concatenate([np.array(f_ch, dtype=float), acf, band_powers])
                f_all.append(f_ch)

            f_all = np.concatenate(f_all, axis=0)

            # Cross-channel features: correlations & covariances (lag 0)
            if C > 1:
                cov = np.cov(Y.T, ddof=1) if w > 1 else np.zeros((C, C))
                corr = np.corrcoef(Y.T) if (w > 1 and np.all(np.std(Y, axis=0) > 0)) else np.eye(C)
                # Take upper triangle without diagonal
                cov_vals = [float(cov[i, j]) for (i, j) in pairs]
                corr_vals = [float(corr[i, j]) for (i, j) in pairs]
                f_all = np.concatenate([f_all, np.array(cov_vals + corr_vals, dtype=float)])

            feats.append(f_all)

            # Build names only once
            if names_once is None:
                names = []
                base_names = ["mean", "median", "std", "iqr", "range", "skew", "kurt", "slope", "pct_out_mad"]
                acf_names = [f"acf_{i+1}" for i in range(fc.acf_lags)]
                bp_names = [f"bp_{i+1}" for i in range(len(fc.bands or []))]
                for c in range(C):
                    names += [f"ch{c}_{n}" for n in (base_names + acf_names + bp_names)]
                if C > 1:
                    for (i, j) in pairs:
                        names.append(f"cov_ch{i}_ch{j}")
                    for (i, j) in pairs:
                        names.append(f"corr_ch{i}_ch{j}")
                names_once = names

        self.feature_names_ = names_once
        return np.vstack(feats)

    # ---------- HMM Core (Gaussian diag) ----------

    def _scale(self, F: np.ndarray, fit: bool = True) -> torch.Tensor:
        if fit or (self.scaler_ is None):
            self.scaler_ = RobustScaler()
            Fz = self.scaler_.fit_transform(F)
        else:
            Fz = self.scaler_.transform(F)
        return torch.as_tensor(Fz, dtype=torch.float32, device=self.device)

    def _init_params(self, Fz: torch.Tensor) -> None:
        T, d = Fz.shape
        km = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(Fz.cpu().numpy())
        mu = []
        var = []
        for k in range(self.k):
            Xk = Fz[labels == k]
            if Xk.shape[0] == 0:
                Xk = Fz[torch.randint(0, T, (max(1, T // self.k),), device=self.device)]
            mu.append(torch.mean(Xk, dim=0))
            v = torch.var(Xk, dim=0, unbiased=False) + 1e-3
            var.append(v)

        self.mu_ = torch.stack(mu, dim=0).to(self.device)
        self.var_ = torch.stack(var, dim=0).to(self.device)
        self.pi_ = torch.full((self.k,), 1.0 / self.k, device=self.device)
        A = torch.full((self.k, self.k), 1.0 / self.k, device=self.device)
        A = 0.85 * torch.eye(self.k, device=self.device) + 0.15 * A
        A = A / A.sum(dim=1, keepdim=True)
        self.A_ = A

    def _log_gaussian_diag(self, X: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        T, d = X.shape
        X2 = X.unsqueeze(1)     # (T, 1, d)
        mu2 = mu.unsqueeze(0)   # (1, k, d)
        var2 = var.unsqueeze(0) # (1, k, d)
        logdet = torch.log(var).sum(dim=1)  # (k,)
        quad = ((X2 - mu2) ** 2 / var2).sum(dim=2)  # (T, k)
        const = d * np.log(2 * np.pi)
        return -0.5 * (const + logdet.unsqueeze(0) + quad)

    def _log_emission(self, X: torch.Tensor) -> torch.Tensor:
        return self._log_gaussian_diag(X, self.mu_, self.var_)

    def _forward_log(self, logB: torch.Tensor) -> Tuple[torch.Tensor, float]:
        T, k = logB.shape
        logA = torch.log(self.A_ + 1e-32)
        logpi = torch.log(self.pi_ + 1e-32)
        alpha = torch.empty((T, k), device=self.device)
        alpha[0] = logpi + logB[0]
        for t in range(1, T):
            alpha[t] = logB[t] + torch.logsumexp(alpha[t - 1].unsqueeze(1) + logA, dim=0)
        loglik = float(torch.logsumexp(alpha[-1], dim=0).item())
        return alpha, loglik

    def _backward_log(self, logB: torch.Tensor) -> torch.Tensor:
        T, k = logB.shape
        logA = torch.log(self.A_ + 1e-32)
        beta = torch.empty((T, k), device=self.device)
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            beta[t] = torch.logsumexp(logA + (logB[t + 1] + beta[t + 1]).unsqueeze(0), dim=1)
        return beta

    def _em_train(self, X: torch.Tensor) -> None:
        prev_ll = -np.inf
        self.loglik_.clear()
        T, d = X.shape
        for it in range(self.max_iter):
            logB = self._log_emission(X)
            alpha, ll = self._forward_log(logB)
            beta = self._backward_log(logB)
            self.loglik_.append(ll)

            log_gamma = alpha + beta
            log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.exp(log_gamma)

            logA = torch.log(self.A_ + 1e-32)
            xi = alpha[:-1].unsqueeze(2) + logA.unsqueeze(0) + logB[1:].unsqueeze(1) + beta[1:].unsqueeze(1)
            xi = xi - torch.logsumexp(xi.view(T - 1, -1), dim=1, keepdim=True).view(T - 1, 1, 1)
            xi = torch.exp(xi)

            self.pi_ = gamma[0] / gamma[0].sum()

            A_num = xi.sum(dim=0)
            A_den = A_num.sum(dim=1, keepdim=True)
            self.A_ = (A_num / (A_den + 1e-12))

            wsum = gamma.sum(dim=0)
            mu_new = (gamma.T @ X) / (wsum.unsqueeze(1) + 1e-12)
            diff = X.unsqueeze(1) - mu_new.unsqueeze(0)
            var_new = (gamma.unsqueeze(2) * (diff ** 2)).sum(dim=0) / (wsum.unsqueeze(1) + 1e-12)
            var_new = torch.clamp(var_new, min=1e-4, max=1e3)

            self.mu_ = mu_new
            self.var_ = var_new

            if it > 0 and abs(ll - prev_ll) < self.tol * (1 + abs(prev_ll)):
                break
            prev_ll = ll

    def viterbi(self, X: torch.Tensor) -> np.ndarray:
        logB = self._log_emission(X)
        T, k = logB.shape
        logA = torch.log(self.A_ + 1e-32)
        logpi = torch.log(self.pi_ + 1e-32)

        delta = torch.empty((T, k), device=self.device)
        psi = torch.empty((T, k), dtype=torch.long, device=self.device)

        delta[0] = logpi + logB[0]
        psi[0] = -1
        for t in range(1, T):
            vals = delta[t - 1].unsqueeze(1) + logA
            psi[t] = torch.argmax(vals, dim=0)
            delta[t] = torch.max(vals, dim=0).values + logB[t]

        zT = torch.argmax(delta[-1]).item()
        z = [zT]
        for t in range(T - 1, 0, -1):
            zT = psi[t, zT].item()
            z.append(zT)
        z.reverse()
        return np.array(z, dtype=int)

    # ---------- Anomaly/Slices ----------

    def _choose_anomaly_state(self, X: torch.Tensor, z: np.ndarray) -> None:
        logB = self._log_emission(X).detach().cpu().numpy()
        ll_state = []
        for k in range(self.k):
            mask = z == k
            if not np.any(mask):
                ll_state.append(np.inf)
            else:
                ll_state.append(-float(logB[mask, k].mean()))
        s_ll = int(np.argmax(ll_state))

        counts = np.bincount(z, minlength=self.k)
        s_small = int(np.argmin(counts))

        if self.anomaly_strategy == "lowest_ll":
            self.anomaly_state_ = s_ll
        elif self.anomaly_strategy == "smallest_state":
            self.anomaly_state_ = s_small
        else:
            self.anomaly_state_ = s_ll if counts[s_ll] <= counts[s_small] else s_small

    def _build_slices(self, z: np.ndarray) -> List[List[int]]:
        if len(z) == 0:
            return []
        slices = []
        s = z[0]
        start = 0
        for i in range(1, len(z)):
            if z[i] != s:
                slices.append([start, i - 1, s])
                s = z[i]
                start = i
        slices.append([start, len(z) - 1, s])
        return slices

    def _enforce_min_dwell(self, slices: List[List[int]]) -> List[List[int]]:
        if not slices:
            return slices
        out = []
        i = 0
        while i < len(slices):
            a, b, s = slices[i]
            length = b - a + 1
            if length >= self.min_slice:
                out.append([a, b, s])
                i += 1
            else:
                if len(out) > 0:
                    out[-1][1] = b
                elif i + 1 < len(slices):
                    na, nb, ns = slices[i + 1]
                    slices[i + 1] = [a, nb, ns]
                else:
                    out.append([a, b, s])
                i += 1
        return out


print("Updated: UnsupervisedStateSegmenter now supports multi-channel (multivariate) series.\n")
