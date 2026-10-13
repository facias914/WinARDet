from typing import Dict, List

import torch
from torch import nn
from transformers import Trainer

from loss import build_coord_token_info, scale_aware_coord_soft_loss
from models.compression import ObjectMemoryCompressor
from utils.constants import IGNORE_INDEX
from utils.rope2d import get_rope_index_25


def attach_segment_context_compressor(model: nn.Module, data_args) -> nn.Module:
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(
        getattr(model.config, "text_config", None), "hidden_size", None
    )
    if hidden_size is None:
        raise ValueError("Cannot infer hidden_size for ObjectMemoryCompressor.")

    compressor_hidden_dim = getattr(data_args, "compressor_hidden_dim", None) or hidden_size
    if compressor_hidden_dim != hidden_size:
        raise ValueError(
            "Segment context compression requires compressor_hidden_dim == model hidden_size "
            f"({compressor_hidden_dim} != {hidden_size})."
        )

    model.context_compressor = ObjectMemoryCompressor(
        hidden_size=hidden_size,
        num_memory_tokens=getattr(data_args, "num_memory_tokens", 4),
        num_heads=getattr(data_args, "compressor_num_heads", 8),
        num_layers=getattr(data_args, "compressor_num_layers", 6),
        passthrough_tokens_per_side=getattr(
            data_args, "compressor_passthrough_tokens_per_side", 0
        ),
        expected_segment_length=getattr(data_args, "max_tokens_per_segment", 300),
    )
    ref_param = next(model.parameters())
    model.context_compressor.to(device=ref_param.device, dtype=ref_param.dtype)
    model.context_compression_config = data_args
    return model


class SegmentContextCompressionTrainer(Trainer):
    """Segment-as-sample trainer with standard causal attention.

    For each sample:
    1) Compress history answer segments into memory tokens.
    2) Build model input as [prefix, history_memory, current_segment].
    3) Train with regular 2D attention mask so flash-attn-2 stays available.
    """

    def __init__(self, *args, tokenizer=None, context_compression_args=None, **kwargs):
        trainer_args = kwargs.get("args")
        if trainer_args is None and len(args) >= 2:
            trainer_args = args[1]
        if trainer_args is not None and getattr(trainer_args, "remove_unused_columns", None):
            trainer_args.remove_unused_columns = False
        super().__init__(*args, **kwargs)
        self.context_tokenizer = tokenizer or self.processing_class
        self.context_compression_args = context_compression_args
        self.coord_token_ids = None
        self.coord_token_start = None
        self._coord_soft_stats = {}
        if getattr(self.context_compression_args, "use_coord_soft_loss", False):
            self.coord_token_ids, self.coord_token_start = build_coord_token_info(
                self.context_tokenizer,
                getattr(self.context_compression_args, "coord_vocab_size", 1000),
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        compressor_owner = getattr(model, "module", model)
        if not hasattr(compressor_owner, "context_compressor"):
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        outputs = self.forward_segment_causal(model, inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def _get_token_embedding(self, model):
        return model.get_input_embeddings()

    def _embed_prefix_with_images(self, model, input_ids, prefix_end, pixel_values, image_grid_thw):
        token_embedding = self._get_token_embedding(model)
        prefix_ids = input_ids[:, :prefix_end]
        prefix_embeds = token_embedding(prefix_ids)

        image_token_id = self.context_tokenizer.convert_tokens_to_ids("<|image_pad|>")

        pixel_values = pixel_values.to(device=prefix_embeds.device, dtype=model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = image_embeds.to(device=prefix_embeds.device, dtype=prefix_embeds.dtype)
        image_mask = (prefix_ids == image_token_id).unsqueeze(-1).expand_as(prefix_embeds)
        prefix_embeds = prefix_embeds.masked_scatter(image_mask, image_embeds)
        return prefix_embeds

    def _compress_history_segments(self, compressor, token_embedding, answer_token_ids, device):
        max_tokens_per_segment = getattr(self.context_compression_args, "max_tokens_per_segment", 300)
        detach_inputs = getattr(self.context_compression_args, "detach_compressor_inputs", True)
        history_no_grad = getattr(self.context_compression_args, "history_compression_no_grad", True)

        segments: List[torch.Tensor] = [
            answer_token_ids[start : start + max_tokens_per_segment]
            for start in range(0, answer_token_ids.numel(), max_tokens_per_segment)
        ]

        sync_loss = None
        hidden_size = token_embedding.embedding_dim
        compressor_dtype = next(compressor.parameters()).dtype

        if len(segments) == 0:
            if not history_no_grad:
                dummy_embeds = torch.zeros(
                    1,
                    max_tokens_per_segment,
                    hidden_size,
                    device=device,
                    dtype=compressor_dtype,
                )
                dummy_mask = torch.ones(
                    1,
                    max_tokens_per_segment,
                    device=device,
                    dtype=torch.bool,
                )
                dummy_memory = compressor(
                    segment_token_embeds=dummy_embeds,
                    segment_attention_mask=dummy_mask,
                )[0]
                sync_loss = dummy_memory.sum() * 0.0
            return [], sync_loss

        incomplete_lengths = [int(seg.numel()) for seg in segments if int(seg.numel()) != max_tokens_per_segment]
        if incomplete_lengths:
            raise ValueError(
                "Segment history compression only supports complete history segments "
                f"of {max_tokens_per_segment} tokens, got {incomplete_lengths}."
            )

        def compress_segment_batch(segment_list: List[torch.Tensor], *, dummy: bool = False):
            if dummy:
                seg_embeds = torch.zeros(
                    1,
                    max_tokens_per_segment,
                    hidden_size,
                    device=device,
                    dtype=compressor_dtype,
                )
                seg_mask = torch.ones(
                    1,
                    max_tokens_per_segment,
                    device=device,
                    dtype=torch.bool,
                )
                return compressor(segment_token_embeds=seg_embeds, segment_attention_mask=seg_mask)[0]

            if len(segment_list) == 0:
                return None
            max_len = max(int(seg.numel()) for seg in segment_list)
            batch_embeds = []
            batch_mask = []
            for seg_ids in segment_list:
                seg_embeds = token_embedding(seg_ids.to(device=device, dtype=torch.long))
                if detach_inputs:
                    seg_embeds = seg_embeds.detach()
                pad_len = max_len - seg_embeds.size(0)
                if pad_len > 0:
                    padding = seg_embeds.new_zeros(pad_len, seg_embeds.size(-1))
                    seg_embeds = torch.cat([seg_embeds, padding], dim=0)
                mask = torch.zeros(max_len, device=device, dtype=torch.bool)
                mask[: seg_ids.numel()] = True
                batch_embeds.append(seg_embeds)
                batch_mask.append(mask)
            batch_embeds = torch.stack(batch_embeds, dim=0)
            batch_mask = torch.stack(batch_mask, dim=0)
            return compressor(segment_token_embeds=batch_embeds, segment_attention_mask=batch_mask)[0]

        if history_no_grad:
            with torch.no_grad():
                memory_batch = compress_segment_batch(segments)
            memory_tokens = memory_batch.detach().reshape(1, -1, memory_batch.size(-1))
        else:
            # One trainable compressor call per sample compresses every historical
            # segment. This keeps all history trainable while avoiding rank-dependent
            # numbers of compressor backward hooks.
            memory_batch = compress_segment_batch(segments)
            memory_tokens = memory_batch.reshape(1, -1, memory_batch.size(-1))

        return [memory_tokens], sync_loss

    def forward_segment_causal(self, model, inputs: Dict):
        input_ids = inputs["input_ids"]   # prefix的所有input id + 当前seg的所有input id
        labels = inputs["labels"]         # prefix的所有input id + 当前seg的所有input id 构成的label

        if input_ids.size(0) != 1:
            raise NotImplementedError("Segment context compression currently supports batch_size == 1.")

        if "context_answer_token_ids" not in inputs or "context_segment_start" not in inputs:
            raise ValueError(
                "SegmentContextCompressionTrainer requires context_answer_token_ids and context_segment_start."
            )

        device = input_ids.device
        token_embedding = self._get_token_embedding(model)
        compressor_owner = getattr(model, "module", model)

        label_row = labels[0]    # prefix的所有input id + 当前seg的所有input id 构成的label
        segment_positions = torch.nonzero(label_row != IGNORE_INDEX, as_tuple=False).flatten()   # 当前seg在labels中的位置索引
        if segment_positions.numel() == 0:
            tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            return {"loss": model(**tensor_inputs).loss}

        prefix_end = int(segment_positions[0].item())
        prefix_embeds = self._embed_prefix_with_images(
            model,
            input_ids,
            prefix_end,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
        )

        answer_token_ids = inputs["context_answer_token_ids"][0]   # 所有answer token的label
        if "context_answer_mask" in inputs:
            answer_token_ids = answer_token_ids[inputs["context_answer_mask"][0].bool()]  # 所有answer token的label

        segment_start = int(inputs["context_segment_start"][0].item())
        segment_start = max(0, min(segment_start, int(answer_token_ids.numel())))
        history_token_ids = answer_token_ids[:segment_start]

        historical_memory_list, compressor_sync_loss = self._compress_history_segments(
            compressor_owner.context_compressor,
            token_embedding,
            history_token_ids,
            device,
        )
        historical_memory_tokens = (
            torch.cat(historical_memory_list, dim=1)
            if historical_memory_list
            else prefix_embeds.new_empty(1, 0, prefix_embeds.size(-1))
        )

        segment_ids = label_row[segment_positions].to(device=device, dtype=torch.long)
        segment_embeds = token_embedding(segment_ids).unsqueeze(0)

        input_embeds = torch.cat([prefix_embeds, historical_memory_tokens, segment_embeds], dim=1)

        seq_len = input_embeds.size(1)
        history_len = historical_memory_tokens.size(1)
        seg_start = prefix_embeds.size(1) + history_len

        lm_labels = torch.full((1, seq_len), IGNORE_INDEX, device=device, dtype=torch.long)
        lm_labels[:, seg_start:] = segment_ids.unsqueeze(0)

        coord_soft_tensors = None
        use_coord_soft_loss = getattr(self.context_compression_args, "use_coord_soft_loss", False)
        if use_coord_soft_loss and "coord_mask" in inputs:
            coord_soft_tensors = {
                "coord_mask": torch.zeros((1, seq_len), device=device, dtype=torch.bool),
                "coord_values": torch.zeros((1, seq_len), device=device, dtype=torch.long),
                "coord_roles": torch.zeros((1, seq_len), device=device, dtype=torch.long),
                "box_widths": torch.zeros((1, seq_len), device=device, dtype=torch.float32),
                "box_heights": torch.zeros((1, seq_len), device=device, dtype=torch.float32),
            }
            for key in coord_soft_tensors:
                source = inputs[key][0, segment_positions].to(device=device)
                coord_soft_tensors[key][:, seg_start:] = source.unsqueeze(0)

        lm_attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

        prefix_ids = input_ids[:, :prefix_end].to(device=device, dtype=torch.long)
        prefix_attention_mask = torch.ones_like(prefix_ids, dtype=torch.long, device=device)
        spatial_merge_size = getattr(
            getattr(self.context_compression_args, "image_processor", None),
            "merge_size",
            2,
        )
        prefix_position_ids, _ = get_rope_index_25(
            spatial_merge_size=spatial_merge_size,
            input_ids=prefix_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=prefix_attention_mask,
        )
        suffix_len = history_len + segment_embeds.size(1)
        if suffix_len > 0:
            text_start = prefix_position_ids.max() + 1 if prefix_end > 0 else torch.tensor(
                0, device=device, dtype=prefix_position_ids.dtype
            )
            suffix_position_ids = (
                torch.arange(suffix_len, device=device, dtype=prefix_position_ids.dtype)
                .view(1, 1, -1)
                .expand(3, 1, -1)
                + text_start
            )
            position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)
        else:
            position_ids = prefix_position_ids

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=lm_attention_mask,
            position_ids=position_ids,
            labels=lm_labels,
            use_cache=False,
        )
        loss_ce = outputs.loss
        loss = loss_ce
        self._coord_soft_stats = {}
        if use_coord_soft_loss and coord_soft_tensors is not None:
            loss_coord_soft, coord_soft_stats = scale_aware_coord_soft_loss(
                logits=outputs.logits,
                labels=lm_labels,
                coord_token_ids=self.coord_token_ids,
                coord_token_start=self.coord_token_start,
                coord_vocab_size=getattr(self.context_compression_args, "coord_vocab_size", 1000),
                alpha=getattr(self.context_compression_args, "coord_soft_alpha", 0.05),
                sigma_min=getattr(self.context_compression_args, "coord_soft_sigma_min", 1.0),
                sigma_max=getattr(self.context_compression_args, "coord_soft_sigma_max", 8.0),
                **coord_soft_tensors,
            )
            lambda_coord_soft = getattr(self.context_compression_args, "lambda_coord_soft", 0.3)
            loss = loss + lambda_coord_soft * loss_coord_soft
            self._coord_soft_stats = {
                "loss_coord_soft": float(loss_coord_soft.detach().float().cpu().item()),
                "lambda_coord_soft": float(lambda_coord_soft),
                **coord_soft_stats,
            }

        compressor_anchor = loss.new_zeros(())
        for param in compressor_owner.context_compressor.parameters():
            compressor_anchor = compressor_anchor + param.sum() * 0.0
        loss = loss + compressor_anchor
        if compressor_sync_loss is not None:
            loss = loss + compressor_sync_loss

        return {"loss": loss, "outputs": outputs}
