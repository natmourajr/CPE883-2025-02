# transformer_lm.py – versão aprimorada 
# =============================================================
# Pequeno Language Model autoregressivo baseado em Transformer
# =============================================================
# Melhorias em relação à versão anterior:
# • Treinamento "teacher‑forcing" em TODOS os passos (perda token‑a‑token).
# • Dataset alternativo para retorno "inputs, targets_shifted".
# • Método generate() para geração autoregressiva.
# • Exemplo de uso com ambos os regimes de treino.
# • Fórmula sinusoidal identicamente à Eq. (3) do paper
#   "Attention Is All You Need" (Vaswani et al., 2017).
# =============================================================

import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------------------
# 1. Embeddings ↔ Unembeddings
# -------------------------------------------------------------
class EmbeddingsAndUnembeddings(nn.Module):
    """Token ⇄ hidden‑space converter com codificação posicional.

    - Escala embeddings por ``sqrt(embed_dim)`` (paper original).
    - Interface idêntica ao Vaswani et al. (sin/cos alternados):
        * ``PE(pos, 2k)   =  sin(pos / 10000^{2k/d})``
        * ``PE(pos, 2k+1) =  cos(pos / 10000^{2k/d})``
    - Suporte a embeddings posicionais aprendíveis.
    - Weight‑tying opcional entre embedding e projeção final.
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

        # Projeção de volta para logits
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.unembed.weight = self.token_embedding.weight  # compartilhamento

    # ---------------------------------------------------------
    @staticmethod
    def _generate_sinusoidal_encoding(max_seq_len: int, embed_dim: int) -> torch.Tensor:
        """Tabela sin‑cos (1, max_seq_len, embed_dim) conforme Vaswani.•Eq (3)."""
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position / div_term)  # dim 2k
        pe[:, 1::2] = torch.cos(position / div_term)  # dim 2k+1
        return pe.unsqueeze(0)  # shape (1, L, D)

    # ---------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Retorna hidden states (batch, seq, embed_dim)."""
        batch, seq_len = input_ids.shape

        if self.pos_embedding is None:
            if seq_len > self.positional_encoding.size(1):
                raise ValueError("Sequence length exceeds max_seq_len of fixed table")
        else:
            if seq_len > self.pos_embedding.num_embeddings:
                raise ValueError("Sequence length exceeds learned positional table")

        tok = self.token_embedding(input_ids) * self.scale

        if self.pos_embedding is not None:
            pos_idx = torch.arange(seq_len, device=input_ids.device)
            pos = self.pos_embedding(pos_idx).unsqueeze(0).expand(batch, -1, -1)
        else:
            pos = self.positional_encoding[:, :seq_len, :].expand(batch, -1, -1)

        return self.dropout(tok + pos)

    # ---------------------------------------------------------
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.unembed(hidden_states)


# -------------------------------------------------------------
# 2. Transformer Encoder‑style
# -------------------------------------------------------------
class TransformerModule(nn.Module):
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
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(input_dim, num_heads,
                                               dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, ffn_dim), act_fn, nn.Dropout(dropout),
                    nn.Linear(ffn_dim, input_dim)
                ),
                "norm2": nn.LayerNorm(input_dim),
                "drop": nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask

    # ---------------------------------------------------------
    def train_lm(self,
                 emb: EmbeddingsAndUnembeddings,
                 loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 epochs: int = 1,
                 grad_clip: Optional[float] = 1.0):
        """Treino token‑a‑token (teacher‑forcing)."""
        emb.to(device); self.to(device)
        for ep in range(1, epochs + 1):
            self.train(); emb.train(); total = 0.0
            for inp, target in loader:  # target possui mesmo shape de inp
                inp, target = inp.to(device), target.to(device)
                seq_len = inp.size(1)
                mask = self._causal_mask(seq_len, device)
                hidden = emb(inp)
                encoded = self(hidden, attn_mask=mask)
                logits = emb.decode(encoded)
                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

                optimizer.zero_grad(); loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(self.parameters()) + list(emb.parameters()), grad_clip)
                optimizer.step(); total += loss.item()
            print(f"Epoch {ep}/{epochs} – loss: {total/len(loader):.4f}")

    # ---------------------------------------------------------
    @torch.no_grad()
    def predict_next_token(self,
                           emb: EmbeddingsAndUnembeddings,
                           input_ids: torch.LongTensor,
                           device: torch.device) -> torch.Tensor:
        self.eval(); emb.eval()
        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        mask = self._causal_mask(seq_len, device)
        hidden = emb(input_ids)
        encoded = self(hidden, attn_mask=mask)
        logits = emb.decode(encoded)
        return logits[0, -1, :]

    # ---------------------------------------------------------
    @torch.no_grad()
    def generate(self,
                 emb: EmbeddingsAndUnembeddings,
                 start_tokens: torch.LongTensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 eos_token: Optional[int] = None,
                 device: Optional[torch.device] = None) -> torch.LongTensor:
        """Gera sequência autoregressivamente."""
        if device is None:
            device = next(self.parameters()).device
        self.eval(); emb.eval()
        seq = start_tokens.to(device)
        for _ in range(max_new_tokens):
            logits = self.predict_next_token(emb, seq, device) / temperature
            if top_k is not None:
                vals, idx = torch.topk(logits, top_k)
                probs = torch.full_like(logits, 0.0)
                probs[idx] = torch.softmax(vals, -1)
            else:
                probs = torch.softmax(logits, -1)
            next_token = torch.multinomial(probs, 1)
            seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
            if eos_token is not None and next_token.item() == eos_token:
                break
        return seq


# -------------------------------------------------------------
# 3. Datasets
# -------------------------------------------------------------
class LanguageModelingDataset(Dataset):
    """Retorna (seq[:-1], seq[1:]) para teacher‑forcing."""
    def __init__(self, tokens: torch.Tensor, window: int):
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        for i in range(len(tokens) - window - 1):
            segment = tokens[i : i + window + 1]
            self.inputs.append(segment[:-1])
            self.targets.append(segment[1:])
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# -------------------------------------------------------------
# 4. Demo – treino completo + geração
# -------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dados sintéticos
    vocab, embed_dim, window = 500, 32, 16  # janela = 16 ⇒ modelo vê 15 e prevê 15
    tokens_stream = torch.randint(0, vocab, (3000,))

    dataset = LanguageModelingDataset(tokens_stream, window)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    emb = EmbeddingsAndUnembeddings(vocab, embed_dim, max_seq_len=window)
    model = TransformerModule(embed_dim, num_layers=3, num_heads=4, ffn_dim=128)

    opt = torch.optim.AdamW(list(emb.parameters()) + list(model.parameters()), lr=3e-4)
    crit = nn.CrossEntropyLoss()

    print("Training…")
    model.train_lm(emb, loader, opt, crit, device, epochs=4)

    # Geração de exemplo
    context = tokens_stream[:5].unsqueeze(0)  # prompt inicial (1,5)
    generated = model.generate(emb, context, max_new_tokens=10, top_k=20)
    print("Prompt:", context.tolist()[0])
    print("Generated:", generated.tolist()[0])