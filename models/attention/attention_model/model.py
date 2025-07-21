import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

# =============================================================
# Embeddings + Unembeddings
# =============================================================
class EmbeddingsAndUnembeddings(nn.Module):
    """Token ↔️ hidden space converter with positional encodings.

    * Scales token embeddings by ``sqrt(embed_dim)``   (Vaswani et al., 2017).
    * Can **tie weights** between embedding ↔ unembedding (default ‑ on).
    * Supports fixed sinusoidal or learnable positional embeddings.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 learned_pos: bool = False,
                 tie_weights: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        if learned_pos:
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
            self.register_buffer("positional_encoding", None, persistent=False)
        else:
            pe = self._generate_sinusoidal_encoding(max_seq_len, embed_dim)
            self.register_buffer("positional_encoding", pe, persistent=False)
            self.pos_embedding = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Projection back to logits
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            # Weight tying (pressupposes same shapes)
            self.unembed.weight = self.token_embedding.weight  # share params

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_sinusoidal_encoding(max_seq_len: int, embed_dim: int) -> torch.Tensor:
        """Return (1, max_seq_len, embed_dim) sinusoidal table."""
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return hidden states (batch, seq, embed_dim)."""
        batch, seq_len = input_ids.shape

        # Positional sanity‑check
        if self.pos_embedding is None:  # fixed case
            if seq_len > self.positional_encoding.size(1):
                raise ValueError(
                    f"Sequence length {seq_len} > max_seq_len="
                    f"{self.positional_encoding.size(1)} – either increase "
                    "max_seq_len or use learned_pos=True")
        else:
            if seq_len > self.pos_embedding.num_embeddings:
                raise ValueError("Sequence length exceeds learned positional table")

        tok = self.token_embedding(input_ids) * self.scale  # scaling ✅

        if self.pos_embedding is not None:
            pos_idx = torch.arange(seq_len, device=input_ids.device)
            pos = self.pos_embedding(pos_idx).unsqueeze(0).expand(batch, -1, -1)
        else:
            pos = self.positional_encoding[:, :seq_len, :].expand(batch, -1, -1)

        return self.dropout(tok + pos)

    # ------------------------------------------------------------------
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project (batch, seq, embed_dim) → (batch, seq, vocab_size)."""
        return self.unembed(hidden_states)


# =============================================================
# Generic Transformer stack
# =============================================================
class TransformerModule(nn.Module):
    """Encoder‑style stack **usable on any sequence of vectors**.

    Implements: (SA → Add&Norm → FFN → Add&Norm)×N.
    Support for causal mask generation (autoregressive tasks).
    """

    def __init__(self,
                 input_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(input_dim, num_heads,
                                               dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, ffn_dim), act_fn, nn.Dropout(dropout),
                    nn.Linear(ffn_dim, input_dim)),
                "norm2": nn.LayerNorm(input_dim),
                "drop": nn.Dropout(dropout)
            }))

    # ------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask)
            x = layer["norm1"](x + layer["drop"](attn_out))
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + layer["drop"](ffn_out))
        return x

    # =========================================================
    # Training helpers for autoregressive LM (1‑step ahead)
    # =========================================================
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Return (seq_len, seq_len) upper‑triangular mask with -inf above diag."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ---------------------------------------------------------
    def train_nlp(self,
                  emb: EmbeddingsAndUnembeddings,
                  loader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module,
                  device: torch.device,
                  epochs: int = 1,
                  grad_clip: Optional[float] = 1.0):
        """One‑step next‑token training with **causal mask**."""
        emb.to(device); self.to(device)
        for ep in range(1, epochs+1):
            self.train(); emb.train()
            total = 0.0
            for inp, target in loader:
                inp, target = inp.to(device), target.to(device)
                seq_len = inp.size(1)
                mask = self._causal_mask(seq_len, device)
                hidden = emb(inp)
                encoded = self(hidden, attn_mask=mask)
                logits = emb.decode(encoded)[:, -1, :]  # last step only
                loss = criterion(logits, target)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.parameters()) + list(emb.parameters()), grad_clip)
                optimizer.step()
                total += loss.item()
            print(f"Epoch {ep}/{epochs} – loss: {total/len(loader):.4f}")

    # ---------------------------------------------------------
    @torch.no_grad()
    def predict_next_token(self,
                           emb: EmbeddingsAndUnembeddings,
                           input_ids: torch.LongTensor,
                           device: torch.device) -> torch.Tensor:
        """Return logits (vocab,) for the token immediately after *input_ids*."""
        self.eval(); emb.eval()
        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        mask = self._causal_mask(seq_len, device)
        hidden = emb(input_ids)
        encoded = self(hidden, attn_mask=mask)
        logits = emb.decode(encoded)
        return logits[0, -1, :]


# =============================================================
# Simple Dataset producing (window[:-1], next_token)
# =============================================================
class NextTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, window: int):
        self.inputs = []
        self.targets = []
        for i in range(len(tokens) - window):
            self.inputs.append(tokens[i:i+window])
            self.targets.append(tokens[i+window])
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# =============================================================
# Demo / quick‑test – one‑step prediction
# =============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic token stream
    vocab, embed_dim, seq_len = 500, 32, 12
    tokens_stream = torch.randint(0, vocab, (2000,))

    # Data
    dataset = NextTokenDataset(tokens_stream, seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Modules
    emb = EmbeddingsAndUnembeddings(vocab, embed_dim, max_seq_len=seq_len)
    model = TransformerModule(embed_dim, num_layers=2, num_heads=4, ffn_dim=128)

    opt = torch.optim.Adam(list(emb.parameters()) + list(model.parameters()), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    print("Training...")
    model.train_nlp(emb, loader, opt, crit, device, epochs=3)

    # Predict 1‑step
    context = tokens_stream[:seq_len].unsqueeze(0)  # (1, seq_len)
    logits = model.predict_next_token(emb, context, device)
    print("Next‑token probs (top‑5):",
          torch.softmax(logits, -1).topk(5).indices.tolist())
