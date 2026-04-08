"""CCT-7/3×1 — Compact Convolutional Transformer для CIFAR-100.

Из статьи: "Escaping the Big Data Paradigm with Compact Transformers" (2021).
Ключевая идея: conv-токенизатор вместо patch embedding → работает на малых датасетах.

CCT-7/3×1:
  7  — число transformer-блоков
  3  — kernel_size токенизатора
  1  — число conv-слоёв в токенизаторе

На 32×32: после токенизатора (conv + maxpool stride=2) → 16×16 = 256 токенов.
~3.7M параметров.

Обучение: AdamW, lr=5e-4, wd=3e-2, CosineAnnealingLR, 200 эпох, Mixup=0.4.
"""
import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """Conv-слои + MaxPool → последовательность токенов."""
    def __init__(self, embedding_dim: int, kernel_size: int = 3, n_conv: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(n_conv):
            layers += [
                nn.Conv2d(in_ch, embedding_dim, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True),
            ]
            in_ch = embedding_dim
        # Финальный MaxPool: 32×32 → 16×16
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)                      # (B, C, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, H*W, C)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SeqPool(nn.Module):
    """Attention-based sequence pooling вместо CLS-токена.
    Позволяет модели самой выбирать какие токены важны."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, tokens, dim)
        w = self.attn(x).softmax(dim=1)  # (B, tokens, 1)
        return (w * x).sum(dim=1)        # (B, dim)


class CCT(nn.Module):
    """CCT-7/3×1 для CIFAR-100 (32×32 input)."""
    def __init__(
        self,
        num_classes:   int   = 100,
        embedding_dim: int   = 256,
        num_heads:     int   = 4,
        num_layers:    int   = 7,
        mlp_ratio:     float = 2.0,
        dropout:       float = 0.1,
        # токенизатор
        kernel_size:   int   = 3,
        n_conv:        int   = 1,
    ):
        super().__init__()
        self.tokenizer = ConvTokenizer(embedding_dim, kernel_size, n_conv)
        # После токенизатора на 32×32: 16×16 = 256 токенов
        seq_len = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        self.dropout   = nn.Dropout(dropout)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm     = nn.LayerNorm(embedding_dim)
        self.seq_pool = SeqPool(embedding_dim)
        self.head     = nn.Linear(embedding_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tokenizer(x)   # (B, 256, 256)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.seq_pool(x)    # (B, 256)
        return self.head(x)
