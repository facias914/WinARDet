import json
import os
import random
from PIL import Image
from tqdm import tqdm

IMAGES_DIR = "/home/u1120250365/share/Rex_Omni/Det/ShipRS/images"
ANNOTATIONS_PATH = "/home/u1120250365/share/Rex_Omni/Det/ShipRS/annotations_rex.json"
OUTPUT_IMAGES_DIR = "/home/u1120250365/share/Rex_Omni/Det/ShipRS_mosaic/images_mosaic"
OUTPUT_ANNOTATIONS_PATH = "/home/u1120250365/share/Rex_Omni/Det/ShipRS_mosaic/annotations_mosaic.json"

MOSAIC_SIZE = 1024
SEED = 42

random.seed(SEED)


def load_annotations(path):
    ann_map = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ann_map[item["image_name"]] = item["annotation"]["boxes"]
    return ann_map


def transform_boxes(boxes, quadrant_idx, cell_w, cell_h, orig_w, orig_h):
    """Transform bboxes from original image space to mosaic space.

    Steps:
        1. Scale box coordinates from original image size to cell size.
        2. Add quadrant offset in the mosaic.

    bbox format: [x1, y1, x2, y2] (xyxy)
    quadrant layout:
        0 | 1
        -----
        2 | 3
    """
    offsets = {
        0: (0, 0),
        1: (cell_w, 0),
        2: (0, cell_h),
        3: (cell_w, cell_h),
    }
    ox, oy = offsets[quadrant_idx]
    sx = cell_w / orig_w
    sy = cell_h / orig_h
    transformed = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        new_x1 = x1 * sx + ox
        new_y1 = y1 * sy + oy
        new_x2 = x2 * sx + ox
        new_y2 = y2 * sy + oy
        transformed.append({
            "bbox": [round(new_x1, 2), round(new_y1, 2), round(new_x2, 2), round(new_y2, 2)],
            "phrase": box["phrase"],
        })
    return transformed


def main():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    ann_map = load_annotations(ANNOTATIONS_PATH)
    image_names = list(ann_map.keys())
    print(f"Loaded {len(image_names)} annotated images")

    cell_w = MOSAIC_SIZE // 2
    cell_h = MOSAIC_SIZE // 2
    out_annotations = []

    for idx, anchor_name in enumerate(tqdm(image_names, desc="Creating mosaics")):
        others = [n for n in image_names if n != anchor_name]
        partners = random.sample(others, 3)

        four_images = [anchor_name] + partners
        random.shuffle(four_images)

        mosaic = Image.new("RGB", (MOSAIC_SIZE, MOSAIC_SIZE))
        all_boxes = []

        for q_idx, img_name in enumerate(four_images):
            img = Image.open(os.path.join(IMAGES_DIR, img_name))
            orig_w, orig_h = img.size

            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), Image.BILINEAR)

            row, col = divmod(q_idx, 2)
            mosaic.paste(img, (col * cell_w, row * cell_h))

            boxes = ann_map.get(img_name, [])
            transformed = transform_boxes(boxes, q_idx, cell_w, cell_h, orig_w, orig_h)
            all_boxes.extend(transformed)

        out_name = f"mosaic_{idx:05d}.jpg"
        mosaic.save(os.path.join(OUTPUT_IMAGES_DIR, out_name), quality=95)

        out_annotations.append(json.dumps({
            "image_name": out_name,
            "annotation": {"boxes": all_boxes},
        }, ensure_ascii=False))

    with open(OUTPUT_ANNOTATIONS_PATH, "w") as f:
        f.write("\n".join(out_annotations) + "\n")

    print(f"Done! {len(out_annotations)} mosaic images saved to {OUTPUT_IMAGES_DIR}")
    print(f"Annotations saved to {OUTPUT_ANNOTATIONS_PATH}")


if __name__ == "__main__":
    main()
