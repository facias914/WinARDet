import json

def convert_coco_to_custom_xyxy(coco_json_path, output_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 1. 建立类别 ID 到名称的映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 2. 建立图片 ID 到信息的映射
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 3. 按图片组织标注信息
    results_map = {}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in images:
            continue
            
        image_name = images[image_id]
        
        # 如果是第一次遇到这张图，初始化结构
        if image_name not in results_map:
            results_map[image_name] = {
                "image_name": image_name,
                "annotation": {"boxes": []}
            }
        
        # 4. 执行坐标转换: [x, y, w, h] -> [xmin, ymin, xmax, ymax]
        x, y, w, h = ann['bbox']
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        
        box_info = {
            "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
            "phrase": categories.get(ann['category_id'], "object")
        }
        
        results_map[image_name]["annotation"]["boxes"].append(box_info)

    # 5. 写入输出文件 (每行一个 JSON 对象)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        for item in results_map.values():
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

    print(f"转换完成！已生成 [xmin, ymin, xmax, ymax] 格式的数据。")

# --- 执行 ---
convert_coco_to_custom_xyxy("/home/u1120230285/wcx/Dota2_train_debug/train_200_240/annotations.json", 
                            "/home/u1120230285/wcx/Dota2_train_debug/train_200_240/annotations_rex.json")