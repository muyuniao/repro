"""
从 Adience 原始标注文件 (fold_X_data.txt) 生成 Adience_train.csv 和 Adience_test.csv
用法：
    python build_adience_csv.py \
        --data_dir /home/duomeitinrfx/data/Adience/folds \
        --image_root /home/duomeitinrfx/data/Adience/faces \
        --output_dir /home/duomeitinrfx/data/Adience/csv \
        --test_fold 0

参数说明：
    --data_dir   : 包含 fold_0_data.txt ~ fold_4_data.txt 的目录
    --image_root : Adience 人脸图片根目录（包含 coarse_tilt_aligned_face.xxx 等文件夹）
    --output_dir : 输出 CSV 的目录
    --test_fold  : 用作测试集的 fold 编号 (0-4)，其余 fold 做训练集
"""

import os
import csv
import argparse

# Adience 原始 age 标签 → OrderChain label 的映射
AGE_TO_LABEL = {
    '(0, 2)': 0,
    '2':      0,
    '3':      0,      # 有时标注为单个数字
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(8, 13)': 2,
    '13':     2,
    '(15, 20)': 3,
    '22':     3,
    '23':     3,
    '(25, 32)': 4,
    '29':     4,
    '34':     4,
    '35':     4,
    '(27, 32)': 4,
    '(38, 43)': 5,
    '36':     5,
    '42':     5,
    '45':     5,
    '(38, 42)': 5,
    '(38, 48)': 5,
    '(48, 53)': 6,
    '46':     6,
    '55':     6,
    '(60, 100)': 7,
    '57':     7,
    '58':     7,
}


def parse_fold_file(filepath):
    """解析一个 fold_X_data.txt 文件，返回 (image_path, label) 列表"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第一行是表头
    header = lines[0].strip().split('\t')
    # 找到关键列的索引
    try:
        user_id_idx = header.index('user_id')
        original_image_idx = header.index('original_image')
        face_id_idx = header.index('face_id')
        age_idx = header.index('age')
    except ValueError as e:
        print(f"Warning: 表头解析失败 {filepath}: {e}")
        print(f"表头内容: {header}")
        return samples

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) <= age_idx:
            continue

        age_str = parts[age_idx].strip()
        if age_str == '' or age_str == 'None':
            continue  # 跳过没有年龄标注的样本

        if age_str not in AGE_TO_LABEL:
            print(f"Warning: 未知的 age 值 '{age_str}'，跳过")
            continue

        label = AGE_TO_LABEL[age_str]
        user_id = parts[user_id_idx].strip()
        original_image = parts[original_image_idx].strip()
        face_id = parts[face_id_idx].strip()

        # Adience 图片路径格式: coarse_tilt_aligned_face.{face_id}.{original_image}
        # 存放在 user_id 子文件夹下
        image_filename = f"coarse_tilt_aligned_face.{face_id}.{original_image}"
        image_path = os.path.join(user_id, image_filename)

        samples.append((image_path, label))

    return samples


def main():
    parser = argparse.ArgumentParser(description='构建 Adience CSV 数据文件')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='包含 fold_0_data.txt ~ fold_4_data.txt 的目录')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Adience 人脸图片根目录（用于校验图片是否存在，可选）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出 CSV 的目录')
    parser.add_argument('--test_fold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='用作测试集的 fold 编号 (0-4)，其余做训练集')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_samples = []
    test_samples = []
    
    for fold_id in range(5):
        fold_file = os.path.join(args.data_dir, f'fold_{fold_id}_data.txt')
        if not os.path.exists(fold_file):
            # 也尝试其他常见命名
            fold_file = os.path.join(args.data_dir, f'fold_frontal_{fold_id}_data.txt')
        
        if not os.path.exists(fold_file):
            print(f"Warning: 找不到 fold {fold_id} 的标注文件，跳过")
            continue

        samples = parse_fold_file(fold_file)
        print(f"Fold {fold_id}: 解析到 {len(samples)} 个有效样本")

        if fold_id == args.test_fold:
            test_samples.extend(samples)
        else:
            train_samples.extend(samples)

    # 如果指定了 image_root，校验图片是否存在
    if args.image_root:
        valid_train = []
        for img_path, label in train_samples:
            full_path = os.path.join(args.image_root, img_path)
            if os.path.exists(full_path):
                valid_train.append((img_path, label))
            else:
                pass  # 静默跳过不存在的图片
        print(f"训练集: {len(valid_train)}/{len(train_samples)} 图片存在")
        train_samples = valid_train

        valid_test = []
        for img_path, label in test_samples:
            full_path = os.path.join(args.image_root, img_path)
            if os.path.exists(full_path):
                valid_test.append((img_path, label))
        print(f"测试集: {len(valid_test)}/{len(test_samples)} 图片存在")
        test_samples = valid_test

    # 写出 CSV
    train_csv = os.path.join(args.output_dir, 'Adience_train.csv')
    test_csv = os.path.join(args.output_dir, 'Adience_test.csv')

    for csv_path, samples, split_name in [
        (train_csv, train_samples, '训练'),
        (test_csv, test_samples, '测试')
    ]:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])
            for img_path, label in samples:
                writer.writerow([img_path, label])
        print(f"{split_name}集已保存到: {csv_path} ({len(samples)} 条)")

    # 统计各类别分布
    print("\n===== 类别分布 =====")
    age_groups = [
        "stage0 (0-2)", "stage1 (4-6)", "stage2 (8-13)", "stage3 (15-20)",
        "stage4 (25-32)", "stage5 (38-43)", "stage6 (48-53)", "stage7 (60+)"
    ]
    for split_name, samples in [("训练集", train_samples), ("测试集", test_samples)]:
        print(f"\n{split_name}:")
        label_counts = {}
        for _, label in samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        for i in range(8):
            count = label_counts.get(i, 0)
            print(f"  label={i} {age_groups[i]}: {count}")


if __name__ == '__main__':
    main()
