from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GRUStockEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.proj(h[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerStockEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        return self.proj(h[:, -1, :])


class PatchTransformerStockEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        patch_len: int = 5,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_len = max(int(patch_len), 1)
        self.patch_dim = self.input_dim * self.patch_len
        self.input_proj = nn.Linear(self.patch_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, seq_len, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Unexpected feature dim {feat_dim}, expected {self.input_dim}")
        pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len
        if pad_len > 0:
            pad = x.new_zeros((batch, pad_len, feat_dim))
            x = torch.cat([x, pad], dim=1)
        num_patches = x.size(1) // self.patch_len
        patches = x.reshape(batch, num_patches, self.patch_len * feat_dim)
        return patches, pad_len

    def encode_tokens(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        patches, _pad_len = self.patchify(x)
        tokens = self.input_proj(patches)
        if patch_mask is not None:
            tokens = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        tokens = self.pos_encoder(tokens)
        tokens = self.encoder(tokens)
        return tokens, patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _patches = self.encode_tokens(x)
        pooled = tokens.mean(dim=1)
        return self.proj(pooled)


class SelectiveSSMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.10,
        kernel_size: int = 4,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = max(int(kernel_size), 2)
        self.expansion = max(int(expansion), 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            groups=hidden_dim,
            padding=self.kernel_size - 1,
        )
        self.param_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))
        self.delta_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * self.expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.expansion, hidden_dim),
            nn.Dropout(dropout),
        )

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.depthwise_conv(x.transpose(1, 2))
        return conv[:, :, : x.size(1)].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        u = F.silu(self._causal_conv(u))
        delta_raw, b, c = self.param_proj(h).chunk(3, dim=-1)
        delta = F.softplus(delta_raw + self.delta_bias)
        a = -torch.exp(self.a_log).unsqueeze(0)
        skip = self.skip.unsqueeze(0)

        state = h.new_zeros((h.size(0), self.hidden_dim))
        outputs = []
        for step in range(h.size(1)):
            step_delta = delta[:, step, :]
            decay = torch.exp(a * step_delta)
            step_u = u[:, step, :]
            step_b = torch.tanh(b[:, step, :])
            step_c = torch.tanh(c[:, step, :])
            state = decay * state + step_delta * step_b * step_u
            outputs.append(step_c * state + skip * step_u)
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y * torch.sigmoid(gate))
        x = residual + self.dropout(y)
        return x + self.ffn(self.ffn_norm(x))


class MambaStockEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SelectiveSSMBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(max(int(n_layers), 1))]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.proj(h[:, -1, :])


class MaskedPatchPretrainer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        patch_len: int = 5,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
        mask_ratio: float = 0.40,
    ) -> None:
        super().__init__()
        self.encoder = PatchTransformerStockEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_len=patch_len,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.mask_ratio = float(mask_ratio)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.encoder.patch_dim),
        )

    def _sample_patch_mask(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        n_mask = max(1, int(round(float(num_patches) * self.mask_ratio)))
        noise = torch.rand(batch_size, num_patches, device=device)
        order = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, order[:, :n_mask], True)
        return mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        patches, _pad_len = self.encoder.patchify(x)
        num_patches = patches.size(1)
        patch_mask = self._sample_patch_mask(batch_size=x.size(0), num_patches=num_patches, device=x.device)
        tokens, patches = self.encoder.encode_tokens(x, patch_mask=patch_mask)
        recon = self.decoder(tokens)
        masked_target = patches[patch_mask]
        masked_recon = recon[patch_mask]
        if masked_target.numel() == 0:
            loss = recon.new_tensor(0.0)
        else:
            loss = F.mse_loss(masked_recon, masked_target, reduction="mean")
        stats = {
            "mask_ratio": float(patch_mask.float().mean().item()) if patch_mask.numel() else 0.0,
            "masked_patches": float(patch_mask.sum().item()),
        }
        return loss, stats


class MultiTaskRanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        return_output_dim: int | None = None,
        risk_output_dim: int | None = None,
        dropout: float = 0.10,
        encoder_family: str = "gru",
        return_head_mode: str = "shared",
        context_dim: int = 16,
        state_context: bool = False,
        liquidity_context: bool = False,
        structure_context: bool = False,
        aux_structure_task: bool = False,
        structure_prototype_task: bool = False,
        patch_len: int = 5,
        state_vocab_size: int = 0,
        liquidity_bucket_count: int = 5,
        structure_vocab_size: int = 0,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        encoder_key = str(encoder_family).strip().lower()
        if encoder_key == "ssm":
            encoder_key = "mamba"
        if encoder_key == "gru":
            self.encoder = GRUStockEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder_key == "transformer":
            self.encoder = TransformerStockEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_heads=transformer_heads,
                n_layers=transformer_layers,
                dropout=dropout,
            )
        elif encoder_key == "patch_transformer":
            self.encoder = PatchTransformerStockEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                patch_len=patch_len,
                n_heads=transformer_heads,
                n_layers=transformer_layers,
                dropout=dropout,
            )
        elif encoder_key == "mamba":
            self.encoder = MambaStockEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=transformer_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_family: {encoder_family}")
        self.return_output_dim = int(return_output_dim or output_dim)
        self.risk_output_dim = int(risk_output_dim or 0)
        self.return_head_mode = str(return_head_mode)
        self.liquidity_bucket_count = int(liquidity_bucket_count)
        self.state_context = bool(state_context and int(state_vocab_size) > 0)
        self.liquidity_context = bool(liquidity_context and int(liquidity_bucket_count) > 0)
        self.structure_context = bool(structure_context and int(structure_vocab_size) > 0)
        self.state_missing_idx = int(state_vocab_size)
        self.liquidity_missing_idx = int(liquidity_bucket_count)
        self.structure_missing_idx = int(structure_vocab_size)
        self.aux_structure_task = bool(aux_structure_task and int(structure_vocab_size) > 0)
        self.structure_prototype_task = bool(structure_prototype_task and int(structure_vocab_size) > 0)

        def _build_return_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.return_output_dim),
            )

        if self.return_head_mode == "shared":
            self.return_head = _build_return_head()
            self.return_bucket_heads = None
            self.default_return_head = None
        elif self.return_head_mode == "liquidity_switch":
            self.return_head = None
            self.default_return_head = _build_return_head()
            self.return_bucket_heads = nn.ModuleList([_build_return_head() for _ in range(self.liquidity_bucket_count)])
        else:
            raise ValueError(f"Unsupported return_head_mode: {return_head_mode}")
        self.risk_head = None
        if self.risk_output_dim > 0:
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.risk_output_dim),
            )
        self.structure_head = None
        if self.aux_structure_task:
            self.structure_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, int(structure_vocab_size)),
            )
        self.structure_prototypes = None
        if self.structure_prototype_task:
            self.structure_prototypes = nn.Parameter(torch.randn(int(structure_vocab_size), hidden_dim) * 0.02)

        context_input_dim = 0
        self.state_embedding = None
        self.liquidity_embedding = None
        self.structure_embedding = None
        if self.state_context:
            self.state_embedding = nn.Embedding(self.state_missing_idx + 1, int(context_dim))
            context_input_dim += int(context_dim)
        if self.liquidity_context:
            self.liquidity_embedding = nn.Embedding(self.liquidity_missing_idx + 1, int(context_dim))
            context_input_dim += int(context_dim)
        if self.structure_context:
            self.structure_embedding = nn.Embedding(self.structure_missing_idx + 1, int(context_dim))
            context_input_dim += int(context_dim)
        self.context_proj = None
        self.context_norm = None
        if context_input_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(context_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.context_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _prepare_ids(ids: torch.Tensor | None, missing_idx: int) -> torch.Tensor | None:
        if ids is None:
            return None
        return torch.where(ids >= 0, ids, torch.full_like(ids, int(missing_idx)))

    def forward(
        self,
        x: torch.Tensor,
        liquidity_bucket: torch.Tensor | None = None,
        state_id: torch.Tensor | None = None,
        structure_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embedding = self.encoder(x)
        context_vectors = []
        if self.state_embedding is not None:
            prepared = self._prepare_ids(state_id, self.state_missing_idx)
            if prepared is not None:
                context_vectors.append(self.state_embedding(prepared))
        if self.liquidity_embedding is not None:
            prepared = self._prepare_ids(liquidity_bucket, self.liquidity_missing_idx)
            if prepared is not None:
                context_vectors.append(self.liquidity_embedding(prepared))
        if self.structure_embedding is not None:
            prepared = self._prepare_ids(structure_id, self.structure_missing_idx)
            if prepared is not None:
                context_vectors.append(self.structure_embedding(prepared))
        if context_vectors and self.context_proj is not None and self.context_norm is not None:
            context = torch.cat(context_vectors, dim=1)
            embedding = self.context_norm(embedding + self.context_proj(context))
        if self.return_head_mode == "shared":
            assert self.return_head is not None
            return_pred = self.return_head(embedding)
        else:
            assert self.default_return_head is not None
            assert self.return_bucket_heads is not None
            return_pred = self.default_return_head(embedding)
            if liquidity_bucket is not None:
                for bucket_idx, head in enumerate(self.return_bucket_heads):
                    mask = liquidity_bucket == int(bucket_idx)
                    if bool(mask.any().item()):
                        return_pred[mask] = head(embedding[mask])
        if self.risk_head is not None:
            risk_pred = self.risk_head(embedding)
            pred = torch.cat([return_pred, risk_pred], dim=1)
        else:
            pred = return_pred
        structure_logits = self.structure_head(embedding) if self.structure_head is not None else None
        return pred, embedding, structure_logits

    def structure_similarity(self, embedding: torch.Tensor) -> torch.Tensor | None:
        if self.structure_prototypes is None:
            return None
        emb = F.normalize(embedding, dim=1)
        prototypes = F.normalize(self.structure_prototypes.to(dtype=embedding.dtype), dim=1)
        return emb @ prototypes.T
