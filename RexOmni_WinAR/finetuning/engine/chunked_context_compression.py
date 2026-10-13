from typing import Dict, List

import torch
from torch import nn
from transformers import Trainer

from models.compression import ObjectMemoryCompressor
from utils.constants import IGNORE_INDEX


def attach_chunked_context_compressor(model: nn.Module, data_args) -> nn.Module:
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(
        getattr(model.config, "text_config", None), "hidden_size", None
    )
    if hidden_size is None:
        raise ValueError("Cannot infer hidden_size for ObjectMemoryCompressor.")

    compressor_hidden_dim = getattr(data_args, "compressor_hidden_dim", None) or hidden_size
    if compressor_hidden_dim != hidden_size:
        raise ValueError(
            "Chunked context compression requires compressor_hidden_dim == model hidden_size "
            f"({compressor_hidden_dim} != {hidden_size})."
        )

    model.context_compressor = ObjectMemoryCompressor(
        hidden_size=hidden_size,
        num_memory_tokens=getattr(data_args, "num_memory_tokens", 4),
        num_heads=getattr(data_args, "compressor_num_heads", 8),
        num_layers=getattr(data_args, "compressor_num_layers", 2),
    )
    ref_param = next(model.parameters())
    model.context_compressor.to(device=ref_param.device, dtype=ref_param.dtype)
    model.context_compression_config = data_args
    return model


class ChunkedContextCompressionTrainer(Trainer):
    """Train one compact answer chunk per dataset item.

    Historical segment memories are rebuilt under ``torch.no_grad()`` so one
    optimizer step never retains graphs from previous chunks of the same long
    answer.
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
        outputs = self.forward_chunked(model, inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def _get_token_embedding(self, model):
        return model.get_input_embeddings()

    def _embed_prefix_with_images(self, model, input_ids, prefix_end, pixel_values, image_grid_thw):
        token_embedding = self._get_token_embedding(model)
        prefix_ids = input_ids[:, :prefix_end]
        prefix_embeds = token_embedding(prefix_ids)

        image_token_id = getattr(model.config, "image_token_id", None)
        if pixel_values is None or image_grid_thw is None or image_token_id is None:
            return prefix_embeds

        try:
            pixel_values = pixel_values.to(
                device=prefix_embeds.device, dtype=model.visual.dtype
            )
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(
                device=prefix_embeds.device, dtype=prefix_embeds.dtype
            )
            image_mask = (prefix_ids == image_token_id).unsqueeze(-1).expand_as(prefix_embeds)
            prefix_embeds = prefix_embeds.masked_scatter(image_mask, image_embeds)
        except Exception as exc:
            if self.args.local_rank in (-1, 0):
                print(f"[chunked_context_compression] fallback image token embeddings: {exc}")
        return prefix_embeds

    def _build_chunk_attention_mask(
        self,
        seq_len: int,
        prefix_len: int,
        historical_memory_len: int,
        segment_spans: List[Dict[str, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        allow = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        prefix_end = prefix_len
        history_end = prefix_len + historical_memory_len

        if prefix_len > 0:
            allow[:, :prefix_end] = True
        if historical_memory_len > 0:
            allow[:, prefix_end:history_end] = True

        previous_memory_ranges = []
        for span in segment_spans:
            seg_start = span["segment_start"]
            seg_end = span["segment_end"]
            mem_start = span["memory_start"]
            mem_end = span["memory_end"]

            allow[seg_start:seg_end, :] = False
            if prefix_len > 0:
                allow[seg_start:seg_end, :prefix_end] = True
            if historical_memory_len > 0:
                allow[seg_start:seg_end, prefix_end:history_end] = True
            for prev_mem_start, prev_mem_end in previous_memory_ranges:
                allow[seg_start:seg_end, prev_mem_start:prev_mem_end] = True
            seg_causal = torch.tril(
                torch.ones(seg_end - seg_start, seg_end - seg_start, device=device, dtype=torch.bool)
            )
            allow[seg_start:seg_end, seg_start:seg_end] = seg_causal

            if mem_end > mem_start:
                allow[mem_start:mem_end, :] = False
                if prefix_len > 0:
                    allow[mem_start:mem_end, :prefix_end] = True
                if historical_memory_len > 0:
                    allow[mem_start:mem_end, prefix_end:history_end] = True
                for prev_mem_start, prev_mem_end in previous_memory_ranges:
                    allow[mem_start:mem_end, prev_mem_start:prev_mem_end] = True
                allow[mem_start:mem_end, seg_start:seg_end] = True
                allow[mem_start:mem_end, mem_start:mem_end] = True
                previous_memory_ranges.append((mem_start, mem_end))

        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        mask = mask.masked_fill(~allow, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def _build_chunk_position_ids(
        self,
        seq_len: int,
        prefix_len: int,
        historical_memory_len: int,
        segment_spans: List[Dict[str, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Map tokens onto the compressed timeline seen by supervised tokens.

        Previous raw segments are hidden by the attention mask, so they must not
        consume position space for later supervised tokens. A segment's raw tokens
        use the same compressed start as the memory that replaces that segment;
        later segments see only that memory slot, yielding contiguous visible ids.
        """
        position_ids_1d = torch.empty(seq_len, device=device, dtype=torch.long)
        cursor = 0
        if prefix_len > 0:
            position_ids_1d[:prefix_len] = torch.arange(
                prefix_len, device=device, dtype=torch.long
            )
            cursor = prefix_len

        history_end = prefix_len + historical_memory_len
        if historical_memory_len > 0:
            position_ids_1d[prefix_len:history_end] = torch.arange(
                cursor,
                cursor + historical_memory_len,
                device=device,
                dtype=torch.long,
            )
            cursor += historical_memory_len

        for span in segment_spans:
            seg_start = span["segment_start"]
            seg_end = span["segment_end"]
            mem_start = span["memory_start"]
            mem_end = span["memory_end"]
            seg_len = seg_end - seg_start
            mem_len = mem_end - mem_start

            if seg_len > 0:
                position_ids_1d[seg_start:seg_end] = torch.arange(
                    cursor,
                    cursor + seg_len,
                    device=device,
                    dtype=torch.long,
                )
            if mem_len > 0:
                position_ids_1d[mem_start:mem_end] = torch.arange(
                    cursor,
                    cursor + mem_len,
                    device=device,
                    dtype=torch.long,
                )
            cursor += mem_len

        return position_ids_1d.view(1, 1, seq_len).expand(3, 1, seq_len)

    def _compress_segments_no_grad(self, compressor, token_embedding, segments, device):
        memories = []
        with torch.no_grad():
            for segment_ids in segments:
                if segment_ids.numel() == 0:
                    continue
                segment_embeds = token_embedding(
                    segment_ids.to(device=device, dtype=torch.long)
                ).unsqueeze(0)
                segment_memory, _ = compressor(
                    segment_token_embeds=segment_embeds.detach()
                )
                memories.append(segment_memory.detach())
        return memories

    def forward_chunked(self, model, inputs: Dict):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")

        if input_ids.size(0) != 1:
            raise NotImplementedError(
                "Chunked context compression currently supports batch_size == 1."
            )

        if "context_chunk_start" not in inputs or "context_chunk_end" not in inputs:
            raise ValueError(
                "ChunkedContextCompressionTrainer requires context_chunk_start/context_chunk_end."
            )

        device = input_ids.device
        token_embedding = self._get_token_embedding(model)
        compressor_owner = getattr(model, "module", model)

        valid = (
            attention_mask[0].bool()
            if attention_mask is not None and attention_mask.dim() == 2
            else torch.ones_like(input_ids[0], dtype=torch.bool)
        )
        label_row = labels[0]
        answer_positions = torch.nonzero(
            (label_row != IGNORE_INDEX) & valid, as_tuple=False
        ).flatten()
        if answer_positions.numel() == 0:
            return {"loss": model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}).loss}

        prefix_end = int(answer_positions[0].item())
        prefix_embeds = self._embed_prefix_with_images(
            model,
            input_ids,
            prefix_end,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
        )

        max_tokens_per_segment = getattr(
            self.context_compression_args, "max_tokens_per_segment", 300
        )
        answer_token_ids = label_row[answer_positions].to(device=device, dtype=torch.long)
        sample_segments = [
            answer_token_ids[start : start + max_tokens_per_segment]
            for start in range(0, answer_token_ids.numel(), max_tokens_per_segment)
        ]
        if not sample_segments:
            raise ValueError("Chunked context compression found no answer segments.")

        chunk_start = int(inputs["context_chunk_start"][0].item())
        chunk_end = int(inputs["context_chunk_end"][0].item())
        chunk_start = max(0, min(chunk_start, len(sample_segments)))
        chunk_end = max(chunk_start, min(chunk_end, len(sample_segments)))
        chunk_segments = sample_segments[chunk_start:chunk_end]
        if not chunk_segments:
            raise ValueError(
                f"Empty chunk range [{chunk_start}, {chunk_end}) for {len(sample_segments)} segments."
            )

        historical_memory_list = self._compress_segments_no_grad(
            compressor_owner.context_compressor,
            token_embedding,
            sample_segments[:chunk_start],
            device,
        )
        historical_memory_tokens = (
            torch.cat(historical_memory_list, dim=1)
            if historical_memory_list
            else prefix_embeds.new_empty(1, 0, prefix_embeds.size(-1))
        )

        chunk_segment_embeds = [
            token_embedding(segment_ids.to(device=device, dtype=torch.long)).unsqueeze(0)
            for segment_ids in chunk_segments
            if segment_ids.numel() > 0
        ]

        chunk_memory_tokens = []
        compressor_used = False
        for local_idx, segment_embeds in enumerate(chunk_segment_embeds):
            global_idx = chunk_start + local_idx
            should_compress = global_idx < len(sample_segments) - 1
            if should_compress:
                compressor_used = True
                compressor_segment_embeds = segment_embeds
                if getattr(self.context_compression_args, "detach_compressor_inputs", True):
                    compressor_segment_embeds = compressor_segment_embeds.detach()
                segment_memory, _ = compressor_owner.context_compressor(
                    segment_token_embeds=compressor_segment_embeds
                )
            else:
                segment_memory = prefix_embeds.new_empty(1, 0, prefix_embeds.size(-1))
            chunk_memory_tokens.append(segment_memory)

        input_parts = [prefix_embeds, historical_memory_tokens]
        segment_spans = []
        cursor = prefix_embeds.size(1) + historical_memory_tokens.size(1)
        for segment_embeds, segment_memory in zip(
            chunk_segment_embeds, chunk_memory_tokens
        ):
            seg_start = cursor
            seg_end = seg_start + segment_embeds.size(1)
            input_parts.append(segment_embeds)
            cursor = seg_end

            mem_start = cursor
            mem_end = mem_start + segment_memory.size(1)
            input_parts.append(segment_memory)
            cursor = mem_end
            segment_spans.append(
                {
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "memory_start": mem_start,
                    "memory_end": mem_end,
                }
            )

        input_embeds = torch.cat(input_parts, dim=1)
        seq_len = input_embeds.size(1)
        lm_labels = torch.full(
            (1, seq_len), IGNORE_INDEX, device=device, dtype=torch.long
        )
        for span, segment_ids in zip(segment_spans, chunk_segments):
            lm_labels[:, span["segment_start"] : span["segment_end"]] = (
                segment_ids.to(device=device, dtype=torch.long).unsqueeze(0)
            )

        lm_attention_mask = self._build_chunk_attention_mask(
            seq_len=seq_len,
            prefix_len=prefix_embeds.size(1),
            historical_memory_len=historical_memory_tokens.size(1),
            segment_spans=segment_spans,
            device=device,
            dtype=input_embeds.dtype,
        )
        position_ids = self._build_chunk_position_ids(
            seq_len=seq_len,
            prefix_len=prefix_embeds.size(1),
            historical_memory_len=historical_memory_tokens.size(1),
            segment_spans=segment_spans,
            device=device,
        )

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=lm_attention_mask,
            position_ids=position_ids,
            labels=lm_labels,
            use_cache=False,
        )
        loss = outputs.loss

        if not compressor_used:
            compressor_anchor = loss.new_zeros(())
            for param in compressor_owner.context_compressor.parameters():
                compressor_anchor = compressor_anchor + param.sum() * 0.0
            loss = loss + compressor_anchor

        return {"loss": loss, "outputs": outputs}
