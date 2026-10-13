import argparse
import json
import os
import re
import math
import multiprocessing as mp
import time
from PIL import Image
from tqdm import tqdm

DEFAULT_PROMPT = "You are a helpful assistant"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="IDEA-Research/Rex-Omni")
    parser.add_argument("--test_jsonl_path", type=str, required=True)
    parser.add_argument("--image_root_dir", type=str, default="")
    parser.add_argument("--save_path", type=str, default="eval_results.jsonl")
    parser.add_argument("--output_box_format", type=str, default="xyxy", choices=["xyxy", "cxcywh", "xywh", "poly"])
    parser.add_argument("--task_name_override", type=str, default="", help="Override task_name saved in prediction JSONL, e.g. task4_detect_all_in_polygon.")
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--max_model_len", type=int, default=20000)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的总显卡数；当 --gpu_ids 为空时使用 0..num_gpus-1")
    parser.add_argument("--gpu_ids", type=str, default="", help="逗号分隔的物理 GPU ID，例如 0,2,3；设置后优先于 --num_gpus")
    parser.add_argument("--question", type=str, default="Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant")
    return parser.parse_args()

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

def worker_inference(rank, gpu_id, args, data_chunk):
    """
    单个 GPU 进程的工作函数
    """
    # Isolate the physical GPU and compile cache before importing CUDA libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    base_cache_root = os.path.expanduser(
        os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")
    )
    worker_cache_root = os.path.join(
        base_cache_root, "rex_omni_data_parallel", f"gpu_{gpu_id}"
    )
    os.environ["VLLM_CACHE_ROOT"] = worker_cache_root

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise RuntimeError(
            f"Worker {rank} expected exactly one visible GPU after binding physical "
            f"GPU {gpu_id}, but found {visible_gpu_count}."
        )
    torch.cuda.set_device(0)
    print(
        f"进程 {rank} 已启动：物理 GPU {gpu_id} -> 逻辑 cuda:0，"
        f"vLLM cache: {worker_cache_root}",
        flush=True,
    )

    # Each process sees one logical GPU, so tensor parallelism stays at one.
    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="slow",
        limit_mm_per_prompt={"image": 10, "video": 10},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1, # 每张卡独立，所以这里始终为1
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=0.8,
        repetition_penalty=1.05,
        top_k=1,
        temperature=0.0,
        skip_special_tokens=False,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    results = []
    # 3. 执行推理
    for entry in tqdm(data_chunk, desc=f"GPU {gpu_id}/rank{rank}"):
        image_path = entry["image_path"]
        categories = entry["categories"]
        gt = entry["gt"]
        dataset_name = entry.get("dataset_name", "unknown")
        task_name = entry.get("task_name", "detection")
        saved_task_name = args.task_name_override or task_name

        full_image_path = os.path.join(args.image_root_dir, image_path) if args.image_root_dir else image_path
        
        try:
            image = Image.open(full_image_path).convert("RGB")
            w, h = image.size
            
            question = args.question.replace("[OBJ]", ", ".join(categories))
            
            # 构建输入消息
            messages = [
                {"role": "system", "content": args.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # 准备 vLLM 输入
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            
            llm_inputs = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_inputs},
            }

            # ``LLM.generate`` is synchronous.  This wall-clock interval covers
            # vLLM's vision encoder/projector, language-model prefill, AR/MTP
            # forwards, token sampling, KV-cache updates, and output decoding.
            generation_start = time.perf_counter()
            outputs = model.generate([llm_inputs], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            generation_time_seconds = time.perf_counter() - generation_start

            # 解析
            extracted = parse_prediction(output_text, w, h, 0, 0, args.output_box_format, task_name)
            
            prediction = {
                "image_path": image_path,
                "extracted_predictions": extracted,
                "gt": gt.get("gt_mask", gt),
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output_text,
                "task_name": saved_task_name,
                "generation_time_seconds": generation_time_seconds,
            }
            results.append(prediction)

        except Exception as e:
            print(f"GPU {gpu_id}/rank{rank} 处理图片 {image_path} 出错: {e}")

    # 4. 每个进程保存自己的结果，防止内存溢出或进程崩溃导致数据全部丢失
    chunk_save_path = args.save_path.replace(".jsonl", f"_gpu{rank}.jsonl")
    with open(chunk_save_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"GPU {gpu_id}/rank{rank} 推理完成，结果已保存至 {chunk_save_path}")

def main():
    args = get_args()
    
    # 加载总数据
    with open(args.test_jsonl_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    if args.gpu_ids.strip():
        gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    else:
        gpu_ids = list(range(max(1, int(args.num_gpus))))
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"GPU IDs must be non-negative: {gpu_ids}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be unique: {gpu_ids}")
    num_gpus = len(gpu_ids)
    chunk_size = math.ceil(len(all_data) / num_gpus) if all_data else 0
    chunks = [all_data[i : i + chunk_size] for i in range(0, len(all_data), chunk_size)] if chunk_size > 0 else []
    if len(chunks) < num_gpus:
        chunks.extend([[] for _ in range(num_gpus - len(chunks))])

    print(f"总数据量: {len(all_data)}, GPU IDs: {gpu_ids}, 分成 {len(chunks)} 份，每份约 {chunk_size} 条")

    chunk_files = [
        args.save_path.replace(".jsonl", f"_gpu{rank}.jsonl")
        for rank in range(num_gpus)
    ]
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

    # Use spawn so every worker imports CUDA libraries after its GPU is isolated.
    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker_inference, args=(rank, gpu_id, args, chunks[rank]))
        p.start()
        processes.append(p)

    remaining = set(range(num_gpus))
    while remaining:
        for worker_id in list(remaining):
            process = processes[worker_id]
            process.join(timeout=0.5)
            if process.exitcode is None:
                continue
            remaining.remove(worker_id)
            if process.exitcode != 0:
                for other_id in remaining:
                    other_process = processes[other_id]
                    if other_process.is_alive():
                        other_process.terminate()
                for other_id in remaining:
                    processes[other_id].join()
                raise RuntimeError(
                    f"GPU worker {worker_id} failed with exit code "
                    f"{process.exitcode}; remaining workers were stopped."
                )

    missing_chunks = [path for path in chunk_files if not os.path.isfile(path)]
    if missing_chunks:
        raise RuntimeError(f"Missing worker output files: {missing_chunks}")

    # 合并结果
    final_save_path = args.save_path
    with open(final_save_path, "w") as outfile:
        for chunk_file in chunk_files:
            with open(chunk_file, "r") as infile:
                outfile.write(infile.read())
            os.remove(chunk_file)

    print(f"🎉 所有 GPU 推理完成！最终结果已汇总至: {final_save_path}")

if __name__ == "__main__":
    main()
