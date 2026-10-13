import argparse
import json
import math
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from PIL import Image
try:
    from safetensors.torch import load_file
except ModuleNotFoundError:
    load_file = None
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
FINETUNE_ROOT = REPO_ROOT / "finetuning"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(FINETUNE_ROOT))

from engine.segment_context_compression import attach_segment_context_compressor
from utils.rope2d import get_rope_index_25

DEFAULT_PROMPT = "You are a helpful assistant."


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="IDEA-Research/Rex-Omni")
    parser.add_argument("--test_jsonl_path", type=str, required=True)
    parser.add_argument("--image_root_dir", type=str, default="")
    parser.add_argument("--save_path", type=str, default="eval_results_chunked_transformers.jsonl")
    parser.add_argument("--output_box_format", type=str, default="xyxy", choices=["xyxy", "cxcywh", "xywh", "poly"])
    parser.add_argument("--task_name_override", type=str, default="", help="Override task_name saved in prediction JSONL, e.g. task4_detect_all_in_polygon.")
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--max_tokens_per_segment", type=int, default=300)
    parser.add_argument("--num_memory_tokens", type=int, default=4)
    parser.add_argument("--compressor_passthrough_tokens_per_side", type=int, default=0)
    parser.add_argument("--compressor_num_layers", type=int, default=2)
    parser.add_argument("--compressor_num_heads", type=int, default=8)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use when --gpu_ids is not set.")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated physical GPU ids to use, e.g. 0,1,3.")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="Number of worker processes to launch on each selected GPU.")
    parser.add_argument("--question", type=str, default="Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--stop_sequences", type=str, nargs="*", default=["<|im_end|>", "<0><0><0><0>"])
    parser.add_argument("--recompute_compact_kv", action="store_true", default=True)
    parser.add_argument("--no_recompute_compact_kv", dest="recompute_compact_kv", action="store_false")
    return parser.parse_args()


def _load_context_compressor_weights(model, model_path):
    if load_file is None:
        print("[compressor] warning: safetensors is not installed; skip context_compressor weight loading.")
        return 0

    single_path = os.path.join(model_path, "model.safetensors")
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(single_path):
        state = load_file(single_path, device="cpu")
        comp_state = {k[len("context_compressor.") :]: v for k, v in state.items() if k.startswith("context_compressor.")}
        if comp_state:
            model.context_compressor.load_state_dict(comp_state, strict=True)
        return len(comp_state)

    if not os.path.exists(index_path):
        return 0

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    shard_names = sorted({v for k, v in weight_map.items() if k.startswith("context_compressor.")})
    comp_state = {}
    for shard_name in shard_names:
        shard = load_file(os.path.join(model_path, shard_name), device="cpu")
        for key, value in shard.items():
            if key.startswith("context_compressor."):
                comp_state[key[len("context_compressor.") :]] = value
    if comp_state:
        missing, unexpected = model.context_compressor.load_state_dict(comp_state, strict=False)
        if missing or unexpected:
            print(f"[compressor] load_state_dict missing={missing}, unexpected={unexpected}")
    return len(comp_state)


def load_model_and_processor(args):
    target_device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    device_kwargs = {"device_map": {"": target_device}} if torch.cuda.is_available() else {}

    def _load(attn_impl, with_device_map=True):
        kwargs = dict(base_kwargs)
        if with_device_map:
            kwargs.update(device_kwargs)
        model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            attn_implementation=attn_impl,
            **kwargs,
        )
        if torch.cuda.is_available() and not with_device_map:
            model_obj = model_obj.to(target_device)
        return model_obj

    try:
        model = _load("flash_attention_2" if torch.cuda.is_available() else "eager", with_device_map=True)
    except Exception as exc:
        print(f"[model] flash_attention_2/device_map load failed, fallback path: {exc}")
        try:
            model = _load("eager", with_device_map=True)
        except Exception as eager_exc:
            print(f"[model] device_map load failed, fallback to explicit .to(cuda/cpu): {eager_exc}")
            model = _load("eager", with_device_map=False)

    compression_args = SimpleNamespace(
        max_tokens_per_segment=args.max_tokens_per_segment,
        num_memory_tokens=args.num_memory_tokens,
        compressor_passthrough_tokens_per_side=args.compressor_passthrough_tokens_per_side,
        compressor_num_heads=args.compressor_num_heads,
        compressor_num_layers=args.compressor_num_layers,
        compressor_hidden_dim=None,
    )
    attach_segment_context_compressor(model, compression_args)
    loaded = _load_context_compressor_weights(model, args.model_path)
    if loaded == 0:
        print("[compressor] warning: no context_compressor weights found; using initialized compressor weights.")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=False,
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True
    return model, processor


def _model_device(model):
    return next(model.parameters()).device


def _model_dtype(model):
    return next(model.parameters()).dtype


def _move_inputs_to_device(inputs, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _embed_prefix_with_images(model, processor_inputs):
    input_ids = processor_inputs["input_ids"]
    token_embedding = model.get_input_embeddings()
    prefix_embeds = token_embedding(input_ids)

    image_token_id = getattr(model.config, "image_token_id", None)
    pixel_values = processor_inputs.get("pixel_values")
    image_grid_thw = processor_inputs.get("image_grid_thw")
    if image_token_id is None or pixel_values is None or image_grid_thw is None:
        return prefix_embeds

    pixel_values = pixel_values.to(device=prefix_embeds.device, dtype=model.visual.dtype)
    image_grid_thw = image_grid_thw.to(device=prefix_embeds.device)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds = image_embeds.to(device=prefix_embeds.device, dtype=prefix_embeds.dtype)
    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(prefix_embeds)
    return prefix_embeds.masked_scatter(image_mask, image_embeds)


def build_inputs(processor, image, question, args):
    messages = [   ######################
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
                {"type": "text", "text": question},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    input_height = inputs["image_grid_thw"][0][1] * 14 if "image_grid_thw" in inputs else torch.tensor(0)
    input_width = inputs["image_grid_thw"][0][2] * 14 if "image_grid_thw" in inputs else torch.tensor(0)
    return inputs, input_height, input_width, text


def _continuous_position_ids(device, start, length):
    pos = torch.arange(start, start + length, device=device, dtype=torch.long)
    return pos.view(1, 1, -1).expand(3, 1, -1)


def _compact_position_ids(processor, prefix_inputs, suffix_len):
    device = prefix_inputs["input_ids"].device
    prefix_ids = prefix_inputs["input_ids"].to(device=device, dtype=torch.long)
    prefix_attention_mask = prefix_inputs.get("attention_mask")
    if prefix_attention_mask is None:
        prefix_attention_mask = torch.ones_like(prefix_ids, dtype=torch.long, device=device)
    else:
        prefix_attention_mask = prefix_attention_mask.to(device=device, dtype=torch.long)

    spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)
    prefix_position_ids, _ = get_rope_index_25(
        spatial_merge_size=spatial_merge_size,
        input_ids=prefix_ids,
        image_grid_thw=prefix_inputs.get("image_grid_thw"),
        video_grid_thw=prefix_inputs.get("video_grid_thw"),
        attention_mask=prefix_attention_mask,
    )

    if suffix_len <= 0:
        return prefix_position_ids, int(prefix_position_ids.max().item()) + 1

    text_start = prefix_position_ids.max() + 1
    suffix_position_ids = (
        torch.arange(suffix_len, device=device, dtype=prefix_position_ids.dtype)
        .view(1, 1, -1)
        .expand(3, 1, -1)
        + text_start
    )
    position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)
    next_position = int(text_start.item()) + suffix_len
    return position_ids, next_position


def _full_attention_mask(device, seq_len):
    return torch.ones((1, seq_len), device=device, dtype=torch.long)


def _cache_position(device, start, length):
    return torch.arange(start, start + length, device=device, dtype=torch.long)


def _cache_seq_len(past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[2])


def _apply_repetition_penalty(scores, generated_ids, penalty):
    if penalty is None or penalty == 1.0 or not generated_ids:
        return scores
    token_ids = torch.tensor(list(set(generated_ids)), device=scores.device, dtype=torch.long)
    token_scores = scores[:, token_ids]
    token_scores = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
    scores[:, token_ids] = token_scores
    return scores


def _sample_next_token(logits, args, generated_ids):
    scores = logits[:, -1, :].float()
    scores = _apply_repetition_penalty(scores, generated_ids, args.repetition_penalty)
    if args.temperature <= 0:
        return torch.argmax(scores, dim=-1)

    scores = scores / args.temperature
    if args.top_k is not None and args.top_k > 0:
        top_k = min(args.top_k, scores.size(-1))
        kth = torch.topk(scores, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if args.top_p is not None and 0 < args.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > args.top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove = torch.zeros_like(scores, dtype=torch.bool)
        remove.scatter_(1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _compress_segment_ids(model, segment_ids):
    if segment_ids.numel() == 0:
        hidden = model.get_input_embeddings().embedding_dim
        return torch.empty(1, 0, hidden, device=segment_ids.device, dtype=_model_dtype(model))
    segment_embeds = model.get_input_embeddings()(segment_ids.to(dtype=torch.long)).unsqueeze(0)
    memory, _ = model.context_compressor(segment_token_embeds=segment_embeds.detach())
    return memory.detach()


def _build_compact_embeds(prefix_embeds, memory_embeds_list):
    parts = [prefix_embeds]
    parts.extend(memory_embeds_list)
    return torch.cat(parts, dim=1)


def _to_legacy_cache(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if isinstance(past_key_values, tuple):
        return past_key_values
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _as_model_cache(past_key_values):
    if past_key_values is None or hasattr(past_key_values, "update"):
        return past_key_values
    if isinstance(past_key_values, tuple) and DynamicCache is not None:
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def _prefill_compact_kv(model, processor, prefix_inputs, compact_embeds):
    """Prefill compact timeline once: [prefix/image tokens] + [memory tokens]."""
    device = compact_embeds.device
    seq_len = compact_embeds.size(1)
    prefix_len = prefix_inputs["input_ids"].size(1)
    suffix_len = seq_len - prefix_len
    position_ids, next_position = _compact_position_ids(processor, prefix_inputs, suffix_len)
    outputs = model(
        inputs_embeds=compact_embeds,
        attention_mask=_full_attention_mask(device, seq_len),
        position_ids=position_ids,
        cache_position=_cache_position(device, 0, seq_len),
        use_cache=True,
        return_dict=True,
    )
    cache_len = _cache_seq_len(outputs.past_key_values)
    if cache_len != seq_len:
        raise RuntimeError(f"Compact prefill KV length mismatch: cache={cache_len}, compact={seq_len}")
    return _to_legacy_cache(outputs.past_key_values), outputs.logits, next_position


def _append_embeds_to_kv(model, embeds, past_key_values, position_start):
    if embeds.size(1) == 0:
        return past_key_values, None, position_start

    device = embeds.device
    past_len = _cache_seq_len(past_key_values)
    length = embeds.size(1)
    outputs = model(
        inputs_embeds=embeds,
        past_key_values=_as_model_cache(past_key_values),
        attention_mask=_full_attention_mask(device, past_len + length),
        position_ids=_continuous_position_ids(device, position_start, length),
        cache_position=_cache_position(device, past_len, length),
        use_cache=True,
        return_dict=True,
    )
    next_position = position_start + length
    expected_cache_len = past_len + length
    cache_len = _cache_seq_len(outputs.past_key_values)
    if cache_len != expected_cache_len:
        raise RuntimeError(
            f"Compact append KV length mismatch: cache={cache_len}, expected={expected_cache_len}"
        )
    return _to_legacy_cache(outputs.past_key_values), outputs.logits, next_position


def _forward_decode_token(model, token_id, past_key_values, position_id):
    device = token_id.device
    past_len = _cache_seq_len(past_key_values)
    outputs = model(
        input_ids=token_id.view(1, 1),
        past_key_values=_as_model_cache(past_key_values),
        attention_mask=_full_attention_mask(device, past_len + 1),
        position_ids=_continuous_position_ids(device, position_id, 1),
        cache_position=_cache_position(device, past_len, 1),
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, outputs.logits


def _make_stop_ids(processor):
    ids = {processor.tokenizer.eos_token_id, getattr(processor.tokenizer, "im_end_id", None), 151645}
    return {int(x) for x in ids if x is not None}


def _make_stop_token_sequences(processor, stop_sequences):
    if not stop_sequences:
        return []

    encoded = []
    tokenizer = processor.tokenizer
    for seq in stop_sequences:
        if not seq:
            continue
        token_ids = tokenizer.encode(seq, add_special_tokens=False)
        if token_ids:
            encoded.append(tuple(int(token_id) for token_id in token_ids))
    return encoded


def _stop_sequence_hit(processor, generated_token_ids, stop_sequences, stop_token_sequences):
    if not stop_sequences:
        return False

    for stop_token_ids in stop_token_sequences:
        stop_len = len(stop_token_ids)
        if stop_len <= len(generated_token_ids) and tuple(generated_token_ids[-stop_len:]) == stop_token_ids:
            return True

    max_stop_tokens = max((len(ids) for ids in stop_token_sequences), default=1)
    tail_token_count = max(8, max_stop_tokens + 4)
    tail_ids = generated_token_ids[-tail_token_count:]
    text = processor.batch_decode([tail_ids], skip_special_tokens=False)[0]
    return any(seq and seq in text for seq in stop_sequences)


@torch.inference_mode()
def generate_with_chunked_memory(model, processor, prefix_inputs, args):
    if not args.recompute_compact_kv:
        raise NotImplementedError("Only recompute_compact_kv=True is implemented for correctness-first inference.")

    device = _model_device(model)
    prefix_inputs = _move_inputs_to_device(prefix_inputs, device)
    prefix_embeds = _embed_prefix_with_images(model, prefix_inputs)

    memory_embeds_list = []
    generated_segments = []
    current_segment_ids = []
    generated_for_penalty = []
    stop_token_ids = _make_stop_ids(processor)
    stop_token_sequences = _make_stop_token_sequences(processor, args.stop_sequences)

    compact_embeds = _build_compact_embeds(prefix_embeds, memory_embeds_list)
    compact_past_key_values, next_logits, compact_next_position = _prefill_compact_kv(model, processor, prefix_inputs, compact_embeds)
    past_key_values = compact_past_key_values
    next_position = compact_next_position

    for _ in range(args.max_new_tokens):
        next_token = _sample_next_token(next_logits, args, generated_for_penalty)
        token_int = int(next_token.item())
        current_segment_ids.append(next_token.detach().reshape(()))
        generated_for_penalty.append(token_int)

        stop = token_int in stop_token_ids or _stop_sequence_hit(
            processor,
            generated_for_penalty,
            args.stop_sequences,
            stop_token_sequences,
        )
        if stop:
            break

        past_key_values, next_logits = _forward_decode_token(
            model,
            next_token,
            past_key_values,
            next_position,
        )
        next_position += 1

        if len(current_segment_ids) >= args.max_tokens_per_segment:
            segment_tensor = torch.stack(current_segment_ids).to(device)
            generated_segments.append(segment_tensor)
            memory_embeds = _compress_segment_ids(model, segment_tensor)
            memory_embeds_list.append(memory_embeds)
            current_segment_ids = []

            compact_past_key_values, next_logits, compact_next_position = _append_embeds_to_kv(
                model,
                memory_embeds,
                compact_past_key_values,
                compact_next_position,
            )
            past_key_values = compact_past_key_values
            next_position = compact_next_position

    if current_segment_ids:
        generated_segments.append(torch.stack(current_segment_ids).to(device))
    if not generated_segments:
        return ""
    generated_ids = torch.cat(generated_segments, dim=0).unsqueeze(0)
    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


def parse_prediction(
    text, w, h, input_height, input_width, output_box_format, task_name=None
):
    """
    Parse model output text to extract category-wise bounding boxes or keypoints.
    Input format example:
    "<|object_ref_start|>cell phone<|object_ref_end|><|box_start|>[[298, 143, 365, 279]]<|box_end|>, <|object_ref_start|>person<|object_ref_end|><|box_start|>[[1, 10, 419, 636], [263, 18, 419, 298]]<|box_end|><|im_end|>"

    For keypoint detection, expects JSON format with bbox and keypoints.

    Returns:
    {
        'category1': [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],  # for bounding boxes
        'category2': [x1,y1,x2,y2,x3,y3,...],  # for polygons/points
        ...
    }
    or for keypoint detection:
    {
        'category1': [{'bbox': [...], 'keypoints': {...}, 'phrase': '...'}],
        ...
    }
    """
    result = {}
    # Handle keypoint detection tasks
    if task_name == "keypoint":
        return parse_keypoint_prediction(text, w, h)

    # Remove the end marker if present
    text = text.split("<|im_end|>")[0]

    # First, try the standard pattern with complete box_start/box_end pairs
    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    # Process standard matches
    for category, box_text in matches:
        result = _process_box_text(category, box_text, result, w, h, output_box_format)

    # Handle special case: incomplete box_end for single object_ref_start
    # Check if there's exactly one object_ref_start and it might be missing box_end
    object_ref_count = text.count("<|object_ref_start|>")

    if object_ref_count == 1 and not matches:
        # Try to extract incomplete pattern
        incomplete_pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)$"
        incomplete_match = re.search(incomplete_pattern, text, re.DOTALL)

        if incomplete_match:
            category, box_text = incomplete_match.groups()
            print(
                f"Found incomplete prediction for category '{category.strip()}', attempting to extract coordinates..."
            )
            result = _process_box_text(
                category, box_text, result, w, h, output_box_format, is_incomplete=True
            )

    return result


def _bin_to_image_coord(coord_bin, coord_index, w, h):
    coord_norm = int(coord_bin) / 999.0
    return coord_norm * (w if coord_index % 2 == 0 else h)


def _process_box_text(category, box_text, result, w, h, output_box_format="xyxy", is_incomplete=False):
    """
    Helper function to process box text and extract coordinates
    """
    category = category.strip()

    if is_incomplete:
        # For incomplete predictions, be more lenient with parsing
        # Remove any trailing incomplete content that might cause issues
        box_text = box_text.strip()
        # Remove trailing commas or incomplete coordinates
        if box_text.endswith(","):
            box_text = box_text[:-1]

    boxes = []
    if output_box_format == "poly":
        box_matches = re.findall(r"<(\d+)>", box_text)
        poly_coord_count = 8
        for start in range(0, len(box_matches) - poly_coord_count + 1, poly_coord_count):
            try:
                coords = [
                    _bin_to_image_coord(box_matches[start + offset], offset, w, h)
                    for offset in range(poly_coord_count)
                ]
            except (ValueError, IndexError) as e:
                if is_incomplete:
                    print(f"Error parsing incomplete polygon coordinates: {e}")
                else:
                    print(f"Error parsing polygon coordinates: {e}")
                continue
            boxes.append(coords)

        if category not in result:
            result[category] = []
        result[category].extend(boxes)
        return result

    # 按逗号分割多个polygon序列
    polygon_sequences = [seq.strip() for seq in box_text.split(",") if seq.strip()]

    for polygon_seq in polygon_sequences:
        # Find all box tokens in the format <{number}>
        box_pattern = r"<(\d+)>"
        box_matches = re.findall(box_pattern, polygon_seq)

        if not box_matches:
            continue

        # Process coordinates flexibly - handle different numbers of coordinates
        i = 0
        while i < len(box_matches):
            try:
                # Try to determine the number of coordinates for this shape
                # For bounding boxes: 4 coordinates (x0, y0, x1, y1)
                # For points: 2 coordinates (x, y)
                # For polygons: multiple pairs of coordinates

                # First, try to find a complete shape by looking ahead
                coords = []
                j = i

                # For bounding boxes (4 coordinates)
                if j + 3 < len(box_matches):
                    # Check if this looks like a bounding box (4 consecutive coordinates)
                    x0_bin = int(box_matches[j])
                    y0_bin = int(box_matches[j + 1])
                    x1_bin = int(box_matches[j + 2])
                    y1_bin = int(box_matches[j + 3])

                    # Validate that coordinates make sense (x1 > x0, y1 > y0)
                    if x1_bin > x0_bin and y1_bin > y0_bin:
                        # Convert from bins [0, 999] back to normalized coordinates [0, 1]
                        x0_norm = x0_bin / 999.0
                        y0_norm = y0_bin / 999.0
                        x1_norm = x1_bin / 999.0
                        y1_norm = y1_bin / 999.0

                        # Convert normalized coordinates directly to target image size
                        x0 = x0_norm * w
                        y0 = y0_norm * h
                        x1 = x1_norm * w
                        y1 = y1_norm * h

                        coords = [x0, y0, x1, y1]
                        i += 4
                    else:
                        # If not a valid bounding box, try as individual coordinates
                        coords = []
                        while j < len(box_matches):
                            try:
                                coord_bin = int(box_matches[j])
                                coord_norm = coord_bin / 999.0
                                if j % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                                j += 1
                            except (ValueError, IndexError):
                                break
                        i = j
                else:
                    # Handle remaining coordinates as individual points or polygon vertices
                    # For incomplete predictions, try to extract partial coordinates
                    coords = []
                    remaining_coords = len(box_matches) - j

                    if is_incomplete and remaining_coords >= 2:
                        # For incomplete predictions, if we have at least 2 coordinates, try to use them
                        # But ensure we have an even number for valid (x,y) pairs
                        valid_coords_count = (remaining_coords // 2) * 2

                        for k in range(valid_coords_count):
                            try:
                                coord_bin = int(box_matches[j + k])
                                coord_norm = coord_bin / 999.0
                                if k % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                            except (ValueError, IndexError):
                                break
                        i = j + valid_coords_count

                        if is_incomplete and len(coords) == 2:
                            # If we only have 2 coordinates, treat as a point
                            pass
                        elif (
                            is_incomplete and len(coords) >= 4 and len(coords) % 2 == 0
                        ):
                            # If we have 4 or more even coordinates, it could be a valid shape
                            pass
                        else:
                            # Skip invalid coordinate sets
                            coords = []
                    else:
                        # Standard processing for complete predictions
                        while j < len(box_matches):
                            try:
                                coord_bin = int(box_matches[j])
                                coord_norm = coord_bin / 999.0
                                if j % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                                j += 1
                            except (ValueError, IndexError):
                                break
                        i = j

                # Add the coordinates if we found any
                if coords:
                    if is_incomplete:
                        print(
                            f"Extracted {len(coords)} coordinates for incomplete prediction: {coords[:8]}..."
                        )  # Show first 8 coords
                    boxes.append(coords)

            except (ValueError, IndexError) as e:
                if is_incomplete:
                    print(f"Error parsing incomplete box coordinates: {e}")
                else:
                    print(f"Error parsing box coordinates: {e}")
                i += 1
                continue

    # Add boxes for this category
    if category not in result:
        result[category] = []
    result[category].extend(boxes)

    return result


def parse_keypoint_prediction(text, w, h):
    """
    Parse keypoint task JSON output to extract bbox and keypoints.

    Expected format:
    ```json
    {
        "person1": {
            "bbox": " <1> <36> <987> <984> ",
            "keypoints": {
                "nose": " <540> <351> ",
                "left eye": " <559> <316> ",
                "right eye": "unvisible",
                ...
            }
        },
        ...
    }
    ```

    Returns:
    Dict with category as key and list of keypoint instances as value
    """
    import json
    import re

    # Extract JSON content from markdown code blocks
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        # Try to find JSON without markdown
        try:
            # Look for JSON-like structure
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                return {}
        except:
            return {}
    else:
        json_str = json_matches[0]

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing keypoint JSON: {e}")
        return {}

    result = {}

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        # Convert bbox coordinates from bins [0, 999] to absolute coordinates
        if isinstance(bbox, str) and bbox.strip():
            # Parse box tokens from string format like " <1> <36> <987> <984> "
            coord_pattern = r"<(\d+)>"
            coord_matches = re.findall(coord_pattern, bbox)

            if len(coord_matches) == 4:
                try:
                    x0_bin, y0_bin, x1_bin, y1_bin = [
                        int(match) for match in coord_matches
                    ]
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h
                    converted_bbox = [x0, y0, x1, y1]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing bbox coordinates: {e}")
                    continue
            else:
                print(
                    f"Invalid bbox format for {instance_id}: expected 4 coordinates, got {len(coord_matches)}"
                )
                continue
        else:
            print(f"Invalid bbox format for {instance_id}: {bbox}")
            continue

        # Convert keypoint coordinates from bins to absolute coordinates
        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible" or kp_coords is None:
                converted_keypoints[kp_name] = "unvisible"
            elif isinstance(kp_coords, str) and kp_coords.strip():
                # Parse box tokens from string format like " <540> <351> "
                coord_pattern = r"<(\d+)>"
                coord_matches = re.findall(coord_pattern, kp_coords)

                if len(coord_matches) == 2:
                    try:
                        x_bin, y_bin = [int(match) for match in coord_matches]
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h
                        converted_keypoints[kp_name] = [x, y]
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing keypoint coordinates for {kp_name}: {e}")
                        converted_keypoints[kp_name] = "unvisible"
                else:
                    print(
                        f"Invalid keypoint format for {kp_name}: expected 2 coordinates, got {len(coord_matches)}"
                    )
                    converted_keypoints[kp_name] = "unvisible"
            else:
                converted_keypoints[kp_name] = "unvisible"

        # Group by category (assuming instance_id contains category info)
        # Try to extract category from instance_id (e.g., "person1" -> "person")
        category = "keypoint_instance"
        if instance_id:
            # Remove numbers from instance_id to get category
            import re

            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        if category not in result:
            result[category] = []

        result[category].append(
            {
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "phrase": instance_id,  # Use instance_id as phrase
            }
        )

    return result


def load_test_data(test_jsonl_path):
    """
    Load test data from JSONL file.

    Returns:
    List of test entries, each containing:
    {
        'image_name': str,
        'question': str,
        'gt': dict,
        'categories': list,
        'task_name': str,
        'dataset_name': str,
        'visual_prompt': dict (optional, for visual prompt tasks)
    }
    """
    test_data = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    test_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return test_data


def _process_entry(entry, model, processor, args):
    image_path = entry["image_path"]
    # categories = sorted(entry["categories"], key=lambda x: str(x).casefold())
    categories = list(entry["categories"])
    gt = entry["gt"]
    dataset_name = entry.get("dataset_name", "unknown")
    task_name = entry.get("task_name", "detection")
    saved_task_name = args.task_name_override or task_name
    full_image_path = os.path.join(args.image_root_dir, image_path) if args.image_root_dir else image_path
    question = args.question.replace("[OBJ]", ", ".join(categories))

    if not os.path.exists(full_image_path):
        print(f"Warning: Image not found: {full_image_path}")
        return None

    try:
        image = Image.open(full_image_path).convert("RGB")
        w, h = image.size
        prefix_inputs, input_height, input_width, _ = build_inputs(processor, image, question, args)  ##########
        # This interval covers the compressed HF generation path: vision
        # encoder/projector, compact-KV prefill, AR forwards, token sampling,
        # segment compression/KV updates, and final output decoding.
        generation_start = time.perf_counter()
        output = generate_with_chunked_memory(model, processor, prefix_inputs, args)
        generation_time_seconds = time.perf_counter() - generation_start

        print(f"[{dataset_name}]")
        print(f"Question: {question}")
        print(f"Output: {output}")

        extracted_predictions = parse_prediction(
            output,
            w,
            h,
            int(input_height.item()) if torch.is_tensor(input_height) else int(input_height),
            int(input_width.item()) if torch.is_tensor(input_width) else int(input_width),
            args.output_box_format,
            task_name,
        )
        return {
            "image_path": image_path,
            "extracted_predictions": extracted_predictions,
            "gt": gt.get("gt_mask", gt) if isinstance(gt, dict) else gt,
            "question": question,
            "dataset_name": dataset_name,
            "raw_response": output,
            "task_name": saved_task_name,
            "generation_time_seconds": generation_time_seconds,
        }
    except Exception as exc:
        print(f"Processing failed for {image_path}, error is {exc}")
        return {
            "image_path": image_path,
            "extracted_predictions": {},
            "gt": gt.get("gt_mask", gt) if isinstance(gt, dict) else gt,
            "question": question,
            "dataset_name": dataset_name,
            "raw_response": "",
            "task_name": saved_task_name,
        }


def _run_entries(entries, model, processor, args, desc="Processing test entries"):
    predictions = []
    for entry in tqdm(entries, desc=desc):
        prediction = _process_entry(entry, model, processor, args)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def _write_predictions(path, predictions):
    with open(path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _parse_gpu_ids(args):
    if args.gpu_ids.strip():
        return [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    return list(range(max(1, int(args.num_gpus))))


def _worker_inference(worker_id, gpu_id, args, data_chunk, chunk_save_path):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        args.device = f"cuda:{gpu_id}"
    else:
        args.device = "cpu"
    print(f"[Worker {worker_id} | GPU {gpu_id}] started, samples={len(data_chunk)}, device={args.device}")

    model, processor = load_model_and_processor(args)
    predictions = _run_entries(data_chunk, model, processor, args, desc=f"GPU {gpu_id}/W{worker_id}")
    _write_predictions(chunk_save_path, predictions)
    print(f"[Worker {worker_id} | GPU {gpu_id}] saved {len(predictions)} predictions to {chunk_save_path}")


def _merge_worker_outputs(save_path, num_workers):
    with open(save_path, "w", encoding="utf-8") as outfile:
        for worker_id in range(num_workers):
            chunk_file = save_path.replace(".jsonl", f"_worker{worker_id}.jsonl")
            if not os.path.exists(chunk_file):
                print(f"Warning: missing worker output {chunk_file}")
                continue
            with open(chunk_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            os.remove(chunk_file)


def main():
    args = get_args()
    gpu_ids = _parse_gpu_ids(args)
    args.procs_per_gpu = max(1, int(args.procs_per_gpu))
    num_workers = max(1, len(gpu_ids) * args.procs_per_gpu)
    if not (args.start_idx == -1 and args.end_idx == -1):
        args.save_path = args.save_path.replace(".jsonl", f"_{args.start_idx}_{args.end_idx}.jsonl")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Loading test data from: {args.test_jsonl_path}")
    test_data = load_test_data(args.test_jsonl_path)
    print(f"Loaded {len(test_data)} test entries")

    selected_test_data = test_data[args.start_idx :] if args.end_idx == -1 else test_data[args.start_idx : args.end_idx]
    print(
        f"Processing {len(selected_test_data)} entries "
        f"(from index {args.start_idx} to {args.end_idx if args.end_idx != -1 else 'end'}), "
        f"gpu_ids={gpu_ids}, procs_per_gpu={args.procs_per_gpu}, workers={num_workers}"
    )

    if num_workers == 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_ids[0])
            args.device = f"cuda:{gpu_ids[0]}"
        else:
            args.device = "cpu"
        model, processor = load_model_and_processor(args)
        predictions = _run_entries(selected_test_data, model, processor, args)
        print(f"\nSaving predictions to: {args.save_path}")
        _write_predictions(args.save_path, predictions)
        print(f"Saved {len(predictions)} predictions")
        return

    chunk_size = math.ceil(len(selected_test_data) / num_workers) if selected_test_data else 0
    chunks = [
        selected_test_data[start : start + chunk_size]
        for start in range(0, len(selected_test_data), chunk_size)
    ] if chunk_size > 0 else []
    if len(chunks) < num_workers:
        chunks.extend([[] for _ in range(num_workers - len(chunks))])

    worker_gpu_ids = []
    for gpu_id in gpu_ids:
        worker_gpu_ids.extend([gpu_id] * args.procs_per_gpu)

    print(f"Launching {num_workers} workers on GPUs {gpu_ids}, chunk_size≈{chunk_size}")
    mp.set_start_method("spawn", force=True)
    processes = []
    for worker_id, gpu_id in enumerate(worker_gpu_ids):
        chunk_save_path = args.save_path.replace(".jsonl", f"_worker{worker_id}.jsonl")
        process = mp.Process(
            target=_worker_inference,
            args=(worker_id, gpu_id, args, chunks[worker_id], chunk_save_path),
        )
        process.start()
        processes.append(process)

    failed = False
    for worker_id, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            print(f"Worker {worker_id} failed with exit code {process.exitcode}")
            failed = True
    if failed:
        raise RuntimeError("One or more GPU workers failed.")

    _merge_worker_outputs(args.save_path, num_workers)
    print(f"All workers finished. Merged predictions saved to: {args.save_path}")


if __name__ == "__main__":
    main()
