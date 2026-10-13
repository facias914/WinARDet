import json

def organize_log(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 以五个横杠以上的分割线作为初始切分点
    # 注意：根据你的描述，每组通常由 Reward 行开始，到 GT 行结束
    entries = content.split('\n\n')
    
    blocks = {}

    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 4:
            continue
        
        # 寻找包含 "GT:" 的那一行
        gt_line = next((line for line in lines if line.startswith('GT: ')), None)
        
        if gt_line:
            # 将 GT 的字符串内容作为 Key，实现分类
            # 如果担心格式微差，可以先 json.loads 再 json.dumps 归一化，
            # 但直接用字符串通常对这种 log 已经足够。
            if gt_line not in blocks:
                blocks[gt_line] = []
            blocks[gt_line].append(entry)

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for gt_key in blocks:
            # 获取属于同一个 GT 的所有组
            group_list = blocks[gt_key]
            
            for i, group in enumerate(group_list):
                f.write(group + "\n\n") # 写入当前组内容
                
                if (i + 1) % 8 == 0:
                    f.write("\n\n")

            if len(group_list) % 8 != 0:
                f.write("\n")

if __name__ == "__main__":
    organize_log('/home/u1120250365/Rex-Omni/finetuning/work_dirs/grpo/log.txt', '/home/u1120250365/Rex-Omni/finetuning/work_dirs/grpo/log_group.txt')