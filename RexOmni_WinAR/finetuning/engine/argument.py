from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Mixed Loss Parameters for Object Detection
    use_soft_ce_for_coords: bool = field(
        default=False,
        metadata={"help": "Whether to use soft CE loss for coordinate tokens"},
    )
    coord_sigma: float = field(
        default=2.0,
        metadata={"help": "Sigma parameter for soft CE loss"},
    )
    max_tolerance_pixel: int = field(
        default=10,
        metadata={"help": "Maximum pixel tolerance for coordinate soft CE loss"},
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    use_context_compression: bool = field(default=False)
    segment_by: str = field(default="num_objects")
    objects_per_segment: int = field(default=50)
    max_tokens_per_segment: int = field(default=300)
    max_segments_per_forward: int = field(default=6)
    num_memory_tokens: int = field(default=4)
    compressor_passthrough_tokens_per_side: int = field(default=0)
    keep_class_tokens: bool = field(default=True)
    compress_coord_and_format_tokens_only: bool = field(default=True)
    compressor_hidden_dim: Optional[int] = field(default=None)
    compressor_num_layers: int = field(default=2)
    compressor_num_heads: int = field(default=8)
    use_roi_feature: bool = field(default=True)
    use_box_embedding: bool = field(default=True)
    use_class_embedding: bool = field(default=True)
    detach_compressor_inputs: bool = field(default=True)
    history_compression_no_grad: bool = field(default=False)
    compression_train_mode: str = field(default="gt_segments")
    inference_compression_mode: str = field(default="periodic")
    lambda_cls: float = field(default=0.0)
    lambda_cov: float = field(default=0.0)
    use_coord_soft_loss: bool = field(default=False)
    lambda_coord_soft: float = field(default=0.3)
    coord_soft_alpha: float = field(default=0.05)
    coord_soft_sigma_min: float = field(default=1.0)
    coord_soft_sigma_max: float = field(default=8.0)
    coord_vocab_size: int = field(default=1000)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    config: str = field(default="vicuna_7b/llava_v15_vicuna7B_pretrain.py")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    context_compressor_lr: Optional[float] = None
