import os

# 1. 填入你存放官方 5 个 txt 文件 (如 fold_0_data.txt 等) 的目录
raw_txt_dir = r'/home/duomeitinrfx/data/Aidence/'
# 2. 转换后生成新文件的输出目录（即最后要在 main.py 里填的 data_root）
output_dir = r'/home/duomeitinrfx/data/Aidence/folds/'

# Adience 官方原始文本到 8 个分类标签的映射
age_map = {
    '(0, 2)': 0, '(4, 6)': 1, '(8, 12)': 2, '(15, 20)': 3,
    '(25, 32)': 4, '(38, 43)': 5, '(48, 53)': 6, '(60, 100)': 7
}

def process_fold(fold_idx):
    lines = []
    file_path = os.path.join(raw_txt_dir, f'fold_{fold_idx}_data.txt')
    with open(file_path, 'r') as f:
        f.readline()  # 跳过第一行的表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            user_id = parts[0]
            img_name = parts[1]
            face_id = parts[2]
            age = parts[3]
            
            # 过滤掉无法确定严格 8 分类的异常数据 (如单独的 '35' 或 'None')
            if age in age_map:
                label = age_map[age]
                # Adience 的 aligned 文件夹图片命名规则通常如下：
                rel_path = f"{user_id}/landmark_aligned_face.{face_id}.{img_name}"
                # 按照原作者读取规则( img_path = item[:-3] ), 组装为: "路径 标签\n"
                lines.append(f"{rel_path} {label}\n")
    return lines

# 读取 5 个原始文件
folds_data = [process_fold(i) for i in range(5)]

# 生成交叉验证所需的特定文件夹结构与文本
for test_fold in range(5):
    fold_dir = os.path.join(output_dir, f'test_fold_is_{test_fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    test_lines = folds_data[test_fold]
    train_lines = []
    for i in range(5):
        if i != test_fold:
            train_lines.extend(folds_data[i])
            
    with open(os.path.join(fold_dir, 'age_test.txt'), 'w') as f:
        f.writelines(test_lines)
        
    with open(os.path.join(fold_dir, 'age_train.txt'), 'w') as f:
        f.writelines(train_lines)

print("划分完成！文件已生成至:", output_dir)