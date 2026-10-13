import os
import re
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_pred_boxes(box_str, img_size=640):
    """解析 <56><301> 格式并缩放到 img_size"""
    nums = re.findall(r'<(\d+)>', box_str)
    coords = [int(n) for n in nums]
    boxes = []
    for i in range(0, len(coords), 4):
        if i + 3 < len(coords):
            box = [c * img_size / 1000 for c in coords[i:i+4]]
            boxes.append(box) 
    return boxes

def draw_on_canvas(boxes, labels, color_map, img_size=640):
    """在 640x640 画布上绘图"""
    canvas = np.full((img_size, img_size, 3), 40, dtype=np.uint8) 
    for label, b_list in zip(labels, boxes):
        color = color_map.get(label, (255, 255, 255))
        for box in b_list:
            p1 = (int(box[1]), int(box[0]))
            p2 = (int(box[3]), int(box[2]))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, label[:10], (p1[0], max(15, p1[1]-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return canvas

def visualize_grpo_blocks(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_content = f.read().strip()
        # 注意：如果 8 组为一块，请确保你的 organized 脚本在每 8 组后加了 \n\n\n
        blocks = [b.strip() for b in raw_content.split('\n\n\n') if b.strip()]

    color_palette = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

    for b_idx, block in enumerate(blocks):
        # 提取当前块内所有的组 (每组包含 GT:)
        groups = [g.strip() for g in block.split('\n\n') if "GT:" in g]
        if not groups: continue

        # 1. 提取 GT 字典
        gt_match = re.search(r'GT: (\{.*\})', groups[0])
        if not gt_match: continue
        
        # 兼容处理单引号并加载 JSON
        gt_json = gt_match.group(1).replace("'", '"')
        gt_dict = json.loads(gt_json)
        
        cat_names = list(gt_dict.keys())
        color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(cat_names)}

        # 2. 准备 2x5 布局 (1个 GT + 8个预测 + 1个空白)
        fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor='#1e1e1e')
        axes = axes.flatten()

        # --- 绘制子图 0: Ground Truth ---
        gt_boxes = [v['boxes'] for v in gt_dict.values()]
        gt_canvas = draw_on_canvas(gt_boxes, cat_names, color_map)
        axes[0].imshow(cv2.cvtColor(gt_canvas, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Ground Truth", color='white', fontsize=12)
        axes[0].axis('off')

        # --- 绘制子图 1-8: 8 个采样 Prediction ---
        for p_idx in range(8):
            ax = axes[p_idx + 1] # 从索引 1 开始放预测图
            if p_idx < len(groups):
                g_text = groups[p_idx]
                # 多类别提取正则
                segments = re.findall(
                    r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>.*?<\|box_start\|>(.*?)<\|box_end\|>',
                    g_text,
                    flags=re.S
                )
                
                pred_dict = {name: [] for name in cat_names}
                if segments:
                    for cls_name, box_str in segments:
                        cls_name = cls_name.strip()
                        if cls_name in pred_dict:
                            pred_dict[cls_name].extend(parse_pred_boxes(box_str))

                    # 绘图逻辑
                    pred_boxes_ordered = [pred_dict.get(name, []) for name in cat_names]
                    p_canvas = draw_on_canvas(pred_boxes_ordered, cat_names, color_map)
                    ax.imshow(cv2.cvtColor(p_canvas, cv2.COLOR_BGR2RGB))

                    # 获取 IoU
                    iou_val = re.search(r'box_iou: ([\d\.]+)', g_text)
                    iou_str = iou_val.group(1)[:6] if iou_val else "N/A"
                    ax.set_title(f"Sample {p_idx+1}\nIoU: {iou_str}", color='cyan', fontsize=10)
            
            ax.axis('off')

        # 第 10 个子图 (索引 9) 留空
        axes[9].axis('off')
        
        plt.suptitle(f"GRPO Sampling Analysis (k=8) - Block {b_idx:03d} | Root Class: {cat_names[0]}", 
                     color='white', fontsize=18, y=0.98)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(output_folder, f"block_{b_idx:03d}_k8.png")
        plt.savefig(save_path, facecolor='#1e1e1e', dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # 请根据实际路径修改
    LOG_FILE = '/home/u1120250365/Rex-Omni/finetuning/work_dirs/grpo/log_group.txt'
    OUT_DIR = 'vis_k8'
    visualize_grpo_blocks(LOG_FILE, OUT_DIR)