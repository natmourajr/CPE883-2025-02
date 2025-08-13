import torch
from dataclasses import dataclass
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np
from scipy.signal import spectrogram

TensorLike = Union[torch.Tensor, "np.ndarray", list]

@dataclass
class TorchSpecParams:
    fs: float                         # taxa de amostragem (Hz)
    n_fft: int = 1024                 # tamanho da FFT
    win_length: Optional[int] = None  # tamanho da janela (default = n_fft)
    hop_length: Optional[int] = None  # salto entre janelas (default = win_length//4)
    window: str = "hann"              # 'hann' | 'hamming' | 'blackman' | 'bartlett'
    center: bool = True               # mesmo comportamento do torch.stft
    pad_mode: str = "reflect"
    normalized: bool = False          # arg do torch.stft (normalização interna)
    scaling: str = "spectrum"         # 'none' | 'spectrum' (1/sum(w^2)) | 'density' (/fs/sum(w^2))
    pre_emphasis: float = 0.0         # 0.0 desliga; típico de áudio ~0.97
    to_db: bool = True                # converte potência em dB
    ref_power: Union[float, str] = "max"  # 'max' ou valor positivo (ex.: 1.0)
    amin: float = 1e-10               # piso numérico antes do log
    top_db: Optional[float] = None    # clipe dinâmico (ex.: 80.0)

def _as_tensor(x: TensorLike, device=None, dtype=None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if dtype is not None:
        x = x.to(dtype)
    if device is not None:
        x = x.to(device)
    return x

def _ensure_bct(x: torch.Tensor) -> torch.Tensor:
    """
    Converte para shape (B, C, T):
      (T,)     -> (1, 1, T)
      (C, T)   -> (1, C, T)
      (B, C, T)-> (B, C, T)
    """
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError("Entrada deve ter shape (T,), (C,T) ou (B,C,T).")

def _make_window(name: str, win_length: int, dtype, device) -> torch.Tensor:
    name = name.lower()
    if name == "hann":
        return torch.hann_window(win_length, periodic=True, dtype=dtype, device=device)
    if name == "hamming":
        return torch.hamming_window(win_length, periodic=True, dtype=dtype, device=device)
    if name == "blackman":
        return torch.blackman_window(win_length, periodic=True, dtype=dtype, device=device)
    if name == "bartlett":
        return torch.bartlett_window(win_length, periodic=True, dtype=dtype, device=device)
    raise ValueError(f"Janela '{name}' não suportada.")

def _pre_emphasize(x: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return x
    y = x.clone()
    # y[..., 0] = x[..., 0]
    y[..., 1:] = x[..., 1:] - alpha * x[..., :-1]
    return y

def _power_to_db(power: torch.Tensor, ref: Union[float, str], amin: float, top_db: Optional[float]) -> torch.Tensor:
    if isinstance(ref, str):
        if ref != "max":
            raise ValueError("ref_power string suportado: apenas 'max'.")
        ref_val = power.amax(dim=(-2, -1), keepdim=True).clamp_min(amin)
    else:
        ref_val = torch.as_tensor(ref, dtype=power.dtype, device=power.device).clamp_min(amin)
    db = 10.0 * torch.log10(power.clamp_min(amin) / ref_val)
    if top_db is not None:
        max_per_ex = db.amax(dim=(-2, -1), keepdim=True)
        db = torch.maximum(db, max_per_ex - top_db)
    return db

def compute_spectrogram_torch(
    x: TensorLike,
    params: TorchSpecParams,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Espectrograma via torch.stft.
    Retorna dict com:
      f: (F,), t: (T_frames,), magnitude: (B,C,F,T), power: (B,C,F,T), db (se to_db=True)
    """
    x = _as_tensor(x, device=device, dtype=dtype)
    x = _ensure_bct(x)  # (B,C,T)

    B, C, T = x.shape
    fs = float(params.fs)
    win_length = params.win_length or params.n_fft
    hop_length = params.hop_length or max(1, win_length // 4)

    # Pré-ênfase
    x = _pre_emphasize(x, params.pre_emphasis)

    # Janela
    window = _make_window(params.window, win_length, dtype=x.dtype, device=x.device)

    # torch.stft opera no último eixo; mantemos (B,C) como batch extra
    # Saída: (B,C, F, T_frames) com return_complex=True
    S = torch.stft(
        x.reshape(-1, T),
        n_fft=params.n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=params.center,
        pad_mode=params.pad_mode,
        normalized=params.normalized,
        onesided=True,
        return_complex=True,
    )
    F_bins, T_frames = S.shape[-2], S.shape[-1]
    S = S.reshape(B, C, F_bins, T_frames)

    magnitude = torch.abs(S)
    power = magnitude ** 2

    # Escalas (aprox. SciPy)
    if params.scaling in ("spectrum", "density"):
        win_power = (window ** 2).sum()
        power = power / win_power
        if params.scaling == "density":
            power = power / fs
    elif params.scaling == "none":
        pass
    else:
        raise ValueError("scaling deve ser 'none' | 'spectrum' | 'density'.")

    # Vetores de frequência e tempo
    f = torch.fft.rfftfreq(params.n_fft, d=1.0 / fs).to(x.device, x.dtype)  # (F,)
    t = torch.arange(T_frames, device=x.device, dtype=x.dtype) * (hop_length / fs)  # (T_frames,)

    out: Dict[str, torch.Tensor] = {
        "f": f,                 # (F,)
        "t": t,                 # (T_frames,)
        "magnitude": magnitude, # (B,C,F,T)
        "power": power,         # (B,C,F,T)
    }

    if params.to_db:
        db = _power_to_db(power, params.ref_power, params.amin, params.top_db)
        out["db"] = db  # (B,C,F,T)

    return out

def make_temp_spec(
    serie: pd.Series,
    win: str = "14D",        # tamanho da janela em TEMPO (ex.: '7D', '14D', '30D')
    hop: str = "12H",        # salto entre janelas (ex.: '6H', '12H', '1D')
    window: str = "hann",
    use_db: bool = True,     # True: retorna também em dB (ref absoluta 1.0)
):
    # 1) Ordenar, converter p/ float e interpolar NaNs (frente-trás)
    x = serie.sort_index().astype("float32").interpolate(limit_direction="both")

    # 2) Estimar fs (Hz) a partir do passo mediano do índice (assume quase regular)
    #    dt em segundos:
    dt_sec = np.median(np.diff(x.index.view("i8"))) / 1e9
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("Não foi possível inferir o passo temporal da série.")
    fs = 1.0 / dt_sec  # Hz

    # 3) Converter janela/salto de tempo -> amostras
    win_sec = pd.Timedelta(win).total_seconds()
    hop_sec = pd.Timedelta(hop).total_seconds()
    win_len = max(8, int(round(win_sec * fs)))
    hop_len = max(1, int(round(hop_sec * fs)))

    # 4) n_fft = próxima potência de 2 >= win_len
    n_fft = 1 << (win_len - 1).bit_length()

    params = TorchSpecParams(
        fs=fs,
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        window=window,
        scaling="density",   # PSD/Hz, mais apropriado p/ sinais ambientais
        pre_emphasis=0.0,
        to_db=use_db,
        ref_power=1.0,       # dB absolutos re 1.0 (ajuste se quiser outra ref)
        top_db=None,         # não clipa a dinâmica
    )

    # 5) Para GPU se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(x.to_numpy())
    out = compute_spectrogram_torch(xt, params, device=device, dtype=torch.float32)

    # 6) Frequência em ciclos/dia (cpd) e eixo de tempo em DATETIME
    f_cpd = (out["f"].cpu().numpy()) * 86400.0                # 1 dia = 86400 s
    t_seconds = out["t"].cpu().numpy()                        # segundos desde o início
    t0 = x.index[0]
    t_datetime = t0 + pd.to_timedelta(t_seconds, unit="s")

    return out, f_cpd, t_datetime, params

# -------------------- Exemplo rápido --------------------
if __name__ == "__main__":
    fs = 1000.0
    Tsec = 2.0
    t = torch.arange(int(fs*Tsec)) / fs
    # Sinal muda de 50 Hz para 150 Hz após 1 s
    x = torch.sin(2*torch.pi*50*t) * (t < 1.0) + torch.sin(2*torch.pi*150*t) * (t >= 1.0)

    params = TorchSpecParams(
        fs=fs,
        n_fft=1024,
        win_length=512,
        hop_length=128,
        window="hann",
        scaling="spectrum",
        pre_emphasis=0.0,
        to_db=True,
        ref_power="max",
        top_db=80.0,
    )

    out = compute_spectrogram_torch(x, params, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Ex.: out["db"].shape == (1,1,F,T)
    # Para visualizar (opcional):
    import matplotlib.pyplot as plt
    plt.pcolormesh(out["t"].cpu(), out["f"].cpu(), out["db"][0,0].cpu(), shading="auto")
    plt.xlabel("Tempo (s)"); plt.ylabel("Frequência (Hz)"); plt.colorbar(label="dB"); plt.show()

def spectrogram_temperature_scipy(
    serie: pd.Series,
    win: str = "14D",          # janela em tempo (ex.: '7D', '14D', '30D')
    hop: str = "12H",          # salto em tempo (ex.: '6H', '12H', '1D')
    window: str = "hann",      # janela do STFT (hann/hamming/...)
    scaling: str = "density",  # 'density' (PSD/Hz) ou 'spectrum'
    detrend: str = "constant", # remove média por segmento
    to_db: bool = True,
    amin: float = 1e-12
):
    """
    Retorna:
      f_cpd: np.ndarray (F,)  frequências em ciclos/dia
      t_dt:  np.ndarray (T,)  tempos como datetime64 (centro de cada janela)
      Z:     np.ndarray (F, T) PSD (un^2/Hz) ou dB (se to_db=True)
      meta:  dict com parâmetros efetivos
    """
    s = serie.sort_index().astype("float32").dropna()
    # Interpola faltantes (frente-trás) se necessário:
    s = s.reindex(s.index.union(pd.date_range(s.index.min(), s.index.max(), freq=pd.infer_freq(s.index) or None)))
    s = s.interpolate(limit_direction="both")

    if len(s) < 4:
        raise ValueError("Série muito curta para espectrograma.")

    # Estimar fs (Hz) a partir do passo mediano
    dt_sec = np.median(np.diff(s.index.view("i8"))) / 1e9
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        raise ValueError("Não foi possível inferir o passo temporal da série.")
    fs = 1.0 / dt_sec  # Hz

    # Converter janela/salto -> amostras
    win_len = int(round(pd.Timedelta(win).total_seconds() * fs))
    hop_len = int(round(pd.Timedelta(hop).total_seconds() * fs))
    win_len = max(8, min(win_len, len(s)))               # garante limites válidos
    hop_len = max(1, hop_len)
    noverlap = max(0, min(win_len - hop_len, win_len - 1))

    # SciPy spectrogram
    f, t, Sxx = spectrogram(
        s.to_numpy(),
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=noverlap,
        nfft=None,              # usa nperseg; mude para potência de 2 se quiser
        detrend=detrend,
        scaling=scaling,        # 'density' → PSD/Hz
        mode="psd",             # retorna potência espectral
    )

    # Frequência em ciclos/dia
    f_cpd = f * 86400.0

    # Tempo em datetime (centro de cada janela)
    t0 = s.index[0]
    t_dt = t0 + pd.to_timedelta(t, unit="s")

    # Opcional em dB
    if to_db:
        Z = 10.0 * np.log10(np.maximum(Sxx, amin))
    else:
        Z = Sxx

    meta = dict(fs=fs, win_len=win_len, hop_len=hop_len, noverlap=noverlap,
                window=window, scaling=scaling, detrend=detrend, to_db=to_db)

    return f_cpd, t_dt.to_numpy(), Z, meta