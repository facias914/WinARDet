from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class ObjectMemoryCompressor(nn.Module):
    """Compress the middle of a fixed answer-token segment into memory tokens.

    Optional passthrough edge tokens keep the first/last N token embeddings
    unchanged and compress only the middle tokens:
        [front_N, compressed_middle_K, back_N]
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_tokens: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        passthrough_tokens_per_side: int = 0,
        expected_segment_length: Optional[int] = None,
        use_box_embedding: bool = True,
        use_class_embedding: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.num_layers = num_layers
        self.passthrough_tokens_per_side = passthrough_tokens_per_side
        self.expected_segment_length = expected_segment_length

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if passthrough_tokens_per_side < 0:
            raise ValueError("passthrough_tokens_per_side must be non-negative.")
        if expected_segment_length is not None and expected_segment_length <= 0:
            raise ValueError("expected_segment_length must be positive when provided.")
        if (
            expected_segment_length is not None
            and passthrough_tokens_per_side * 2 >= expected_segment_length
        ):
            raise ValueError(
                "passthrough_tokens_per_side leaves no middle tokens to compress "
                f"({passthrough_tokens_per_side} * 2 >= {expected_segment_length})."
            )

        if expected_segment_length is None:
            raise ValueError(
                "expected_segment_length is required for learnable segment-local position embeddings."
            )
        compressed_segment_length = expected_segment_length - 2 * passthrough_tokens_per_side
        if compressed_segment_length < num_memory_tokens:
            raise ValueError(
                "num_memory_tokens cannot exceed the number of compressed segment tokens "
                f"({num_memory_tokens} > {compressed_segment_length})."
            )
        self.segment_position_embeddings = nn.Parameter(
            torch.randn(compressed_segment_length, hidden_size) * 0.02
        )
        self.segment_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.query_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])
        self.kv_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _build_interval_memory_queries(
        self,
        compress_inputs: torch.Tensor,
        compress_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize memory queries by averaging equal token intervals."""
        batch_size, compressed_length, hidden_size = compress_inputs.shape
        if compress_mask is None and compressed_length % self.num_memory_tokens == 0:
            interval_length = compressed_length // self.num_memory_tokens
            return compress_inputs.view(
                batch_size, self.num_memory_tokens, interval_length, hidden_size
            ).mean(dim=2)

        pooled_queries = []
        for memory_idx in range(self.num_memory_tokens):
            start = memory_idx * compressed_length // self.num_memory_tokens
            end = (memory_idx + 1) * compressed_length // self.num_memory_tokens
            if end <= start:
                end = min(start + 1, compressed_length)
            chunk = compress_inputs[:, start:end]
            if compress_mask is None:
                pooled = chunk.mean(dim=1)
            else:
                chunk_mask = compress_mask[:, start:end].to(dtype=chunk.dtype).unsqueeze(-1)
                denom = chunk_mask.sum(dim=1).clamp_min(1.0)
                pooled = (chunk * chunk_mask).sum(dim=1) / denom
            pooled_queries.append(pooled)
        return torch.stack(pooled_queries, dim=1)

    def forward(
        self,
        segment_token_embeds: torch.Tensor,
        segment_attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return memory tokens and class-token embeddings.

        Args:
            segment_token_embeds: Answer segment embeddings, shape [B, L_seg, D].
            segment_attention_mask: Optional bool mask, shape [B, L_seg], True for valid tokens.

        Returns:
            memory_tokens: [B, K, D] when passthrough_tokens_per_side == 0,
                otherwise [B, 2 * passthrough_tokens_per_side + K, D].
            class_token_embeds: [B, 0, D]
        """
        if segment_token_embeds.dim() != 3:
            raise ValueError(
                "ObjectMemoryCompressor expects segment_token_embeds with shape [B, L_seg, D]."
            )

        device = self.segment_position_embeddings.device
        dtype = self.segment_position_embeddings.dtype
        segment_inputs = segment_token_embeds.to(device=device, dtype=dtype)
        segment_mask = None
        if segment_attention_mask is not None:
            segment_mask = segment_attention_mask.to(device=device, dtype=torch.bool)

        batch_size, segment_length, _ = segment_inputs.shape
        expected_length = self.expected_segment_length
        if expected_length is not None:
            if segment_mask is None:
                if segment_length != expected_length:
                    raise ValueError(
                        "ObjectMemoryCompressor received a non-complete segment: "
                        f"length {segment_length}, expected {expected_length}."
                    )
            else:
                valid_lengths = segment_mask.sum(dim=1)
                if not torch.all(valid_lengths == expected_length):
                    raise ValueError(
                        "ObjectMemoryCompressor requires every compressed item to be a complete "
                        f"segment of {expected_length} tokens, got {valid_lengths.tolist()}."
                    )
                segment_inputs = segment_inputs[:, :expected_length]
                segment_mask = segment_mask[:, :expected_length]
                segment_length = expected_length

        edge = self.passthrough_tokens_per_side
        if edge > 0:
            if edge * 2 >= segment_length:
                raise ValueError(
                    "passthrough_tokens_per_side leaves no middle tokens to compress "
                    f"({edge} * 2 >= {segment_length})."
                )
            front_tokens = segment_inputs[:, :edge]
            back_tokens = segment_inputs[:, -edge:]
            compress_inputs = segment_inputs[:, edge:-edge]
            compress_mask = segment_mask[:, edge:-edge] if segment_mask is not None else None
        else:
            front_tokens = None
            back_tokens = None
            compress_inputs = segment_inputs
            compress_mask = segment_mask

        compressed_length = compress_inputs.size(1)
        if compressed_length != self.segment_position_embeddings.size(0):
            raise ValueError(
                "ObjectMemoryCompressor received an unexpected compressed segment length: "
                f"{compressed_length}, expected {self.segment_position_embeddings.size(0)}."
            )
        position_embeddings = self.segment_position_embeddings.to(device=device, dtype=dtype)
        segment_tokens = self.segment_proj(compress_inputs + position_embeddings.unsqueeze(0))
        memory = self._build_interval_memory_queries(compress_inputs, compress_mask)
        key_padding_mask = None
        if compress_mask is not None:
            key_padding_mask = ~compress_mask
        for attn, query_norm, kv_norm, ffn_norm, ffn in zip(
            self.cross_attn_layers,
            self.query_norms,
            self.kv_norms,
            self.ffn_norms,
            self.ffns,
        ):
            # Pre-RMSNorm residual block, following the Qwen2.5-VL ViT pattern.
            norm_segment_tokens = kv_norm(segment_tokens)
            attended, _ = attn(
                query=query_norm(memory),
                key=norm_segment_tokens,
                value=norm_segment_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            memory = memory + attended
            memory = memory + ffn(ffn_norm(memory))

        memory = self.output_proj(self.output_norm(memory))
        if edge > 0:
            memory = torch.cat([front_tokens, memory, back_tokens], dim=1)
        empty_class_tokens = torch.empty(batch_size, 0, self.hidden_size, device=device, dtype=dtype)
        return memory, empty_class_tokens
