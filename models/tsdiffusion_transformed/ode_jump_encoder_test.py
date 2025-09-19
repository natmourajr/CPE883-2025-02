from .ode_jump import ODEJump,ODEFunc
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # (B, T, C)
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,              # (B, T_q, C) — p.ex., saídas do decoder
        kv: torch.Tensor,             # (B, T_kv, C) — p.ex., saídas do encoder
        kv_key_padding_mask=None,     # (B, T_kv) True = PADDING/ignorar
        attn_mask=None                # (T_q, T_kv) ou (B*n_heads, T_q, T_kv)
    ):
        # MHA já faz: attn = softmax(QK^T/sqrt(d_k)) V
        attn_out, attn_weights = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=kv_key_padding_mask,  # opcional
            attn_mask=attn_mask,                   # opcional (p.ex., máscara causal)
            need_weights=True
        )
        # Residual + norm (padrão Transformer)
        x = self.norm(q + self.out(attn_out))
        return x, attn_weights  # attn_weights: (B, n_heads, T_q, T_kv) a partir do PyTorch 2.6+

class JumpODEEncoder(nn.Module):
    """
    Self‑Attentive Jump‑ODE simplificado:
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    - Self‑attention usa máscara para faltantes (opcional).
    """
    def __init__(self, in_dim, hidden_dim, attn_heads=4, num_layers=2, dropout=0.1, ff_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            nhead=attn_heads,
            d_model=hidden_dim,
            norm_first=True,  # normalização antes da atenção
            dropout=dropout,  # dropout opcional
            dim_feedforward= self.hidden_dim * 4 or ff_dim,  # feedforward dimension
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        self.cross_attn = CrossAttention(hidden_dim,attn_heads)
        self.time_encoding = TimeHybridEncoding(hidden_dim)

    @staticmethod
    def _causal_mask(L, device):
        # Máscara triangular superior (impede olhar para o futuro)
        return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)    
        
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
                delta_h = (x[:, i] - h) / dt_i
                f1 = self.odefunc(delta_h)
                f2 = self.odefunc(delta_h + 0.5*dt_i*f1)
                f3 = self.odefunc(delta_h + 0.5*dt_i*f2)
                f4 = self.odefunc(delta_h + dt_i*f3)
                h  = h + (dt_i/6.0)*(f1 + 2*f2 + 2*f3 + f4)
            states.append(h)
            h=x[:, i]
        H = torch.stack(states, dim=1)
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []

        for i in range(T):
            h = self.gru(H[:, i], h)                                 # jump
            states.append(h)

        H = torch.stack(states, dim=1)   # (B, T, hidden_dim)
        tm_e = self.time_encoding(ts)  # tempo contínuo
        z = x + tm_e

        # máscara causal (L,L) para o Transformer (batch_first=True)
        causal = self._causal_mask(T, x.device)  # (T,T) com -inf acima da diagonal
        z = self.transformer(z,mask=causal)
        H,_ = self.cross_attn(H, z) # (B, T, hidden_dim)'''
        return H


class TimeHybridEncoding(nn.Module):
    """
    Odd dims  -> ramp(t) = a*t + b   (aprendido)
    Even dims -> sinusoidal [sin, cos] com frequências log-espalhadas
    Usa timestamps normalizados globalmente: (ts - t0)/TS_SPAN
    """
    def __init__(self, d_model: int,
                 min_period: float = 4.0,
                 max_period: float = 10_000.0,
                 ramp_gain: float = 1.0):
        super().__init__()
        self.d_model = d_model

        # Máscaras pares/ímpares
        even_idx = torch.arange(d_model) % 2 == 0
        odd_idx  = ~even_idx
        self.register_buffer("even_mask", even_idx, persistent=False)
        self.register_buffer("odd_mask",  odd_idx,  persistent=False)

        self.n_even = int(even_idx.sum().item())
        self.n_odd  = int(odd_idx.sum().item())

        # --- Ramp nos ímpares ---
        self.a = nn.Parameter(torch.zeros(self.n_odd, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(self.n_odd, dtype=torch.float32))
        if ramp_gain != 1.0:
            with torch.no_grad():
                self.a.mul_(ramp_gain)

        # --- Sin/cos nos pares ---
        self.n_freq = max(self.n_even // 2, 0)
        if self.n_freq > 0:
            freqs = torch.exp(
                -torch.linspace(0, math.log(max_period/min_period), self.n_freq, dtype=torch.float32)
            ) * (2.0 * math.pi / min_period)
            self.register_buffer("freqs", freqs, persistent=False)
        else:
            self.register_buffer("freqs", torch.empty(0, dtype=torch.float32), persistent=False)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: (B,T) — timestamps já normalizados: (ts - t0)/TS_SPAN
        retorna: (B,T,D)
        """
        B, T = ts.shape
        device, dtype = ts.device, ts.dtype
        out = torch.zeros(B, T, self.d_model, device=device, dtype=ts.dtype)

        # ----- RAMP nas dims ímpares -----
        if self.n_odd > 0:
            ramp = ts.unsqueeze(-1) * self.a.view(1,1,-1) + self.b.view(1,1,-1)  # (B,T,n_odd)
            odd_positions = self.odd_mask.nonzero(as_tuple=False).squeeze(-1)
            out.index_copy_(dim=2, index=odd_positions, source=ramp)

        # ----- SIN/COS nas dims pares -----
        if self.n_even > 0 and self.n_freq > 0:
            even_positions = self.even_mask.nonzero(as_tuple=False).squeeze(-1)
            phase = ts.unsqueeze(-1) * self.freqs.view(1,1,-1)   # (B,T,n_freq)
            S = torch.sin(phase)
            C = torch.cos(phase)
            sc = torch.stack([S, C], dim=-1).reshape(B, T, -1)  # (B,T,2*n_freq)
            n_fill = min(sc.size(-1), self.n_even)
            out[:, :, even_positions[:n_fill]] = sc[:, :, :n_fill]

        return out
    
class ODEJumpEncoder(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float,float] = [0.9, 0.1],
        n_heads: int = 4,
        n_layers: int = 4                       
        ):
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam)  # <- chama ODEJump.__init__
        # (daqui pra baixo você pode sobrescrever/estender o que quiser)
        
        self.encoder_ode = JumpODEEncoder(hidden_dim, hidden_dim, attn_heads=n_heads, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None,
        x_denoised: torch.Tensor = None
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
            h = self.encoder(torch.cat([x, mask] if x_denoised is None else [x,x_denoised,mask], dim=-1))
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        
        h = self.encoder_ode(h, timestamps)
        state = h
        return state,self.decoder(state) if return_x_hat else None
                
                    