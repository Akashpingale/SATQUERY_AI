"""CDVQA constrained-classifier model.

CDVQA answers come from a closed vocabulary (yes/no, land-cover class names, change-ratio
buckets), so the correct, honest formulation is **classification over the answer vocabulary**,
not free-form generation. The model consumes:

    T1 image features  +  T2 image features  +  (T2 - T1) difference  +  question
        -> fused representation -> softmax over the closed answer vocabulary

Image features are precomputed by a frozen encoder (see scripts/train_cd_vqa.py) so training is
tractable on a 4 GB GPU. Only the question encoder and fusion head are trained.
"""

from __future__ import annotations

import torch
from torch import nn


class CDVQAModel(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        num_answers: int,
        image_feat_dim: int = 512,
        embed_dim: int = 128,
        question_dim: int = 256,
        hidden_dim: int = 256,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(word_vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, question_dim, batch_first=True)
        # T1, T2 and the temporal difference => 3 * image_feat_dim.
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(question_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(
        self, question_ids: torch.Tensor, feat_t1: torch.Tensor, feat_t2: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(question_ids)
        _, hidden = self.gru(embedded)
        q_vec = hidden.squeeze(0)  # (B, question_dim)

        img = torch.cat([feat_t1, feat_t2, feat_t2 - feat_t1], dim=-1)
        img_vec = self.image_proj(img)  # (B, hidden_dim)

        fused = torch.cat([q_vec, img_vec], dim=-1)
        return self.classifier(fused)
