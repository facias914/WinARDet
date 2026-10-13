import json

def convert_format(input_file, output_file):
    converted_list = []
    
    # 1. 逐行读取原始 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            # 提取基础信息
            image_name = data.get("image_name", "")
            boxes_info = data.get("annotation", {}).get("boxes", [])
            
            # 2. 构造聚合结构
            # 使用字典来按类别(phrase)收集坐标
            gt_dict = {}
            categories = []
            
            for box_obj in boxes_info:
                phrase = box_obj.get("phrase", "Unknown")
                bbox = box_obj.get("bbox", [])
                
                if phrase not in gt_dict:
                    gt_dict[phrase] = []
                    categories.append(phrase)
                
                # 添加坐标到对应类别下
                gt_dict[phrase].append(bbox)
            
            # 3. 构造目标格式对象
            new_item = {
                "image_path": image_name,
                "categories": categories,
                "gt": gt_dict,
                "dataset_name": "COCO", # 根据你的要求固定
                "task_name": "common_object_detection" # 根据你的要求固定
            }
            
            converted_list.append(new_item)

    # 4. 保存为新的 JSONL 文件（或者标准 JSON 列表）
    # 这里建议保存为 JSONL 以保持和你输入一致的风格
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"转换完成！处理了 {len(converted_list)} 条数据，已保存至 {output_file}")

# --- 执行转换 ---
convert_format('/home/u1120230285/wcx/Dota2_train_debug/train_200_240/annotations_rex.json', 
                '/home/u1120230285/wcx/Dota2_train_debug/train_200_240/annotations_rex1.json')