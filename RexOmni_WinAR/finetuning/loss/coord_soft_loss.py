from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.constants import IGNORE_INDEX


def build_coord_token_info(tokenizer, coord_vocab_size: int = 1000) -> Tuple[torch.Tensor, Optional[int]]:
    """Return coordinate token ids and a contiguous start id when possible."""
    coord_token_ids = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for coord_value in range(coord_vocab_size):
        token = f"<{coord_value}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_token_id:
            raise ValueError(f"Coordinate token {token!r} is missing from tokenizer vocabulary.")
        coord_token_ids.append(int(token_id))

    coord_token_start = coord_token_ids[0]
    is_contiguous = all(
        token_id == coord_token_start + idx for idx, token_id in enumerate(coord_token_ids)
    )
    if not is_contiguous:
        coord_token_start = None

    return torch.tensor(coord_token_ids, dtype=torch.long), coord_token_start


def scale_aware_coord_soft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_mask: torch.Tensor,
    coord_values: torch.Tensor,
    coord_roles: torch.Tensor,
    box_widths: torch.Tensor,
    box_heights: torch.Tensor,
    coord_token_ids: Optional[torch.Tensor] = None,
    coord_token_start: Optional[int] = None,
    coord_vocab_size: int = 1000,
    alpha: float = 0.05,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Scale-aware Gaussian soft-label CE for causal-LM coordinate tokens.

    The function performs the same causal shift as standard LM CE:
    logits[..., :-1, :] predicts labels[..., 1:]. Coordinate metadata is shifted
    in the same way so soft supervision lands on the same target positions.
    """
    empty_stats = {
        "coord_token_count": 0,
        "coord_soft_mean_sigma": 0.0,
        "coord_soft_min_sigma": 0.0,
        "coord_soft_max_sigma": 0.0,
    }
    if logits.size(1) < 2:
        return logits.new_zeros(()), empty_stats

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_coord_mask = coord_mask[..., 1:].to(device=logits.device, dtype=torch.bool)
    shift_coord_values = coord_values[..., 1:].to(device=logits.device, dtype=torch.long)
    shift_coord_roles = coord_roles[..., 1:].to(device=logits.device, dtype=torch.long)
    shift_box_widths = box_widths[..., 1:].to(device=logits.device, dtype=torch.float32)
    shift_box_heights = box_heights[..., 1:].to(device=logits.device, dtype=torch.float32)

    valid = shift_coord_mask & (shift_labels.to(device=logits.device) != IGNORE_INDEX)
    if not torch.any(valid):
        return logits.new_zeros(()), empty_stats

    flat_logits = shift_logits[valid]
    if coord_token_start is not None:
        coord_logits = flat_logits[:, coord_token_start : coord_token_start + coord_vocab_size]
    else:
        if coord_token_ids is None:
            raise ValueError("coord_token_ids is required when coordinate token ids are not contiguous.")
        coord_token_ids = coord_token_ids.to(device=logits.device, dtype=torch.long)
        coord_logits = flat_logits.index_select(dim=-1, index=coord_token_ids)

    coord_values_valid = shift_coord_values[valid].clamp(0, coord_vocab_size - 1).float()
    coord_roles_valid = shift_coord_roles[valid]
    widths_valid = shift_box_widths[valid].clamp_min(0.0)
    heights_valid = shift_box_heights[valid].clamp_min(0.0)

    use_width = (coord_roles_valid == 0) | (coord_roles_valid == 2)
    scale = torch.where(use_width, widths_valid, heights_valid)
    sigma = (scale * alpha).clamp(min=sigma_min, max=sigma_max)

    coord_axis = torch.arange(coord_vocab_size, device=logits.device, dtype=torch.float32)
    distance = coord_axis.unsqueeze(0) - coord_values_valid.unsqueeze(1)
    soft_target = torch.exp(-(distance * distance) / (2.0 * sigma.unsqueeze(1).pow(2)))

    half_span = (scale * 0.5).clamp_min(1.0)
    within_box_side = distance.abs() <= half_span.unsqueeze(1)
    soft_target = soft_target.masked_fill(~within_box_side, 0.0)
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    log_probs = F.log_softmax(coord_logits.float(), dim=-1)
    loss = -(soft_target * log_probs).sum(dim=-1).mean()
    stats = {
        "coord_token_count": int(valid.sum().detach().cpu().item()),
        "coord_soft_mean_sigma": float(sigma.mean().detach().cpu().item()),
        "coord_soft_min_sigma": float(sigma.min().detach().cpu().item()),
        "coord_soft_max_sigma": float(sigma.max().detach().cpu().item()),
    }
    return loss.to(dtype=logits.dtype), stats
