import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD_IDX


class LexicalEmbedding(nn.Module):
    def __init__(self, vocab_size, input_dim, padding_idx=PAD_IDX):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)

    def forward(self, X):
        return self.emb(X.long())


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, padding_idx=PAD_IDX):
        super().__init__()
        self.padding_idx = padding_idx
        self.position_embeddings = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        with torch.no_grad():
            self.position_embeddings.weight[0].zero_()

    def forward(self, X, input_ids):
        # PAD positions are excluded from position counting, non-PAD positions are numbered from 1
        mask = (input_ids != self.padding_idx).long()
        position_ids = torch.cumsum(mask, dim=1) * mask
        return X + self.position_embeddings(position_ids)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        var  = X.var(dim=-1, correction=0, keepdim=True)
        return (X - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class MLP(nn.Module):
    # I am using SwiGLU from week10 instead of Relu.
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        inner_dim = int(hidden_dim * 2 / 3)
        self.gate_proj   = nn.Linear(input_dim, inner_dim)
        self.value_proj  = nn.Linear(input_dim, inner_dim)
        self.output_proj = nn.Linear(inner_dim, input_dim)
        self.silu        = nn.SiLU()

    def forward(self, X):
        return self.output_proj(self.silu(self.gate_proj(X)) * self.value_proj(X))


class MultiHeadAttention(nn.Module):
    # All heads are computed in parallel via matrix multiplication with reshaping.
    # causal=True applies an upper-triangular mask to prevent attending to future tokens.
    def __init__(self, input_dim, q_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_dim   = q_dim
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

    def split_heads(self, X):
        # (batch, seq_len, input_dim) → (batch, n_heads, seq_len, q_dim)
        return X.unflatten(-1, (self.n_heads, self.q_dim)).transpose(-3, -2)

    def forward(self, X, causal=False, key_padding_mask=None):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.q_dim)

        if causal:
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)

        if key_padding_mask is not None:
            # unsqueeze twice to broadcast across heads and query positions
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out  = attn @ V
        # merge heads: (batch, n_heads, seq_len, q_dim) → (batch, seq_len, input_dim)
        out  = out.transpose(-3, -2).flatten(-2)
        return self.W_o(out)


class MultiHeadCrossAttention(nn.Module):
    # Q from the decoder, K and V from the encoder output.
    # No causal mask here; decoder attends freely to all encoder positions.
    def __init__(self, input_dim, q_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_dim   = q_dim
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

    def split_heads(self, X):
        # (batch, seq_len, input_dim) → (batch, n_heads, seq_len, q_dim)
        return X.unflatten(-1, (self.n_heads, self.q_dim)).transpose(-3, -2)

    def forward(self, decoder_x, encoder_output, src_padding_mask=None):
        Q = self.split_heads(self.W_q(decoder_x))
        K = self.split_heads(self.W_k(encoder_output))
        V = self.split_heads(self.W_v(encoder_output))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.q_dim)

        if src_padding_mask is not None:
            # unsqueeze twice to broadcast across heads and decoder positions
            scores = scores.masked_fill(src_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out  = attn @ V
        # merge heads: (batch, n_heads, tgt_len, q_dim) → (batch, tgt_len, input_dim)
        out  = out.transpose(-3, -2).flatten(-2)
        return self.W_o(out)


class EncoderLayer(nn.Module):
    # Two sublayers:
    # 1. bi-directional multi-head self-attention
    # 2. MLP
    def __init__(self, input_dim, q_dim, n_heads, mlp_hidden_dim, dropout_p=0.1):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)
        self.attn  = MultiHeadAttention(input_dim, q_dim, n_heads)
        self.mlp   = MLP(input_dim, mlp_hidden_dim)
        self.drop  = nn.Dropout(dropout_p)

    def forward(self, X, src_padding_mask=None):
        X = X + self.drop(self.attn(self.norm1(X), causal=False, key_padding_mask=src_padding_mask))
        X = X + self.drop(self.mlp(self.norm2(X)))
        return X


class DecoderLayer(nn.Module):
    # Three sublayers:
    # 1. masked multi-head self-attention (only attends to previous decoder outputs)
    # 2. multi-head cross-attention (Q from decoder, K/V from encoder)
    # 3. MLP
    def __init__(self, input_dim, q_dim, n_heads, mlp_hidden_dim, dropout_p=0.1):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)
        self.norm3 = LayerNorm(input_dim)
        self.self_attn  = MultiHeadAttention(input_dim, q_dim, n_heads)
        self.cross_attn = MultiHeadCrossAttention(input_dim, q_dim, n_heads)
        self.mlp        = MLP(input_dim, mlp_hidden_dim)
        self.drop       = nn.Dropout(dropout_p)

    def forward(self, X, encoder_output, src_padding_mask=None):
        X = X + self.drop(self.self_attn(self.norm1(X), causal=True))
        X = X + self.drop(self.cross_attn(self.norm2(X), encoder_output, src_padding_mask))
        X = X + self.drop(self.mlp(self.norm3(X)))
        return X


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, input_dim, q_dim, n_heads, mlp_hidden_dim,
                 n_enc_layers, n_dec_layers, max_len=256,
                 padding_idx=PAD_IDX, dropout_p=0.1):
        super().__init__()
        self.padding_idx = padding_idx

        self.src_emb    = LexicalEmbedding(vocab_size, input_dim, padding_idx)
        self.src_pos    = LearnedPositionalEmbedding(input_dim, max_len, padding_idx)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(input_dim, q_dim, n_heads, mlp_hidden_dim, dropout_p)
            for _ in range(n_enc_layers)
        ])

        self.tgt_emb    = LexicalEmbedding(vocab_size, input_dim, padding_idx)
        self.tgt_pos    = LearnedPositionalEmbedding(input_dim, max_len, padding_idx)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(input_dim, q_dim, n_heads, mlp_hidden_dim, dropout_p)
            for _ in range(n_dec_layers)
        ])

        self.lm_head = nn.Linear(input_dim, vocab_size)

    def encode(self, src_ids):
        src_padding_mask = (src_ids == self.padding_idx)
        X = self.src_pos(self.src_emb(src_ids), src_ids)
        for layer in self.enc_layers:
            X = layer(X, src_padding_mask)
        return X, src_padding_mask

    def decode(self, tgt_ids, encoder_output, src_padding_mask):
        X = self.tgt_pos(self.tgt_emb(tgt_ids), tgt_ids)
        for layer in self.dec_layers:
            X = layer(X, encoder_output, src_padding_mask)
        return X

    def forward(self, src_ids, tgt_ids):
        encoder_output, src_padding_mask = self.encode(src_ids)
        decoder_output = self.decode(tgt_ids, encoder_output, src_padding_mask)
        return self.lm_head(decoder_output)  # (batch, tgt_len, vocab_size)
