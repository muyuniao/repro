import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# 1. 设定你的 Adience 原始数据集路径
adience_root = r"/home/duomeitinrfx/data/Aidence" # 请替换为你的真实路径
txt_files = glob.glob(os.path.join(adience_root, "fold_*_data.txt"))

# 2. 定义年龄区间到 label 的映射字典
age_map = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,  
    '(8, 13)': 2,  
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 42)': 5,
    '(38, 43)': 5, 
    '(48, 53)': 6,
    '(60, 100)': 7
}

data_list = []

# 3. 解析 txt 文件提取所有有效数据
for txt_file in txt_files:
    df_raw = pd.read_csv(txt_file, sep='\t')
    
    for index, row in df_raw.iterrows():
        age = str(row['age'])
        
        if age in age_map:
            label = age_map[age]
            user_id = str(row['user_id'])
            face_id = str(row['face_id'])
            original_image = str(row['original_image'])
            
            img_name = f"coarse_tilt_aligned_face.{face_id}.{original_image}"
            image_path = os.path.join(adience_root, "faces", user_id, img_name)
            
            if os.path.exists(image_path):
                data_list.append({
                    'image_path': image_path,
                    'label': label
                })

df_all = pd.DataFrame(data_list)

# 4. 按照 75% 训练集和 25% 测试集进行分层切分
train_df, test_df = train_test_split(
    df_all, 
    test_size=0.25, 
    random_state=42, 
    stratify=df_all['label'] # 确保训练集和测试集中的年龄类别分布一致
)

# 5. 分别保存为两个 CSV 文件
train_csv = os.path.join(adience_root, "Adience_train.csv")
test_csv = os.path.join(adience_root, "Adience_test.csv")

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"数据切分完成！")
print(f"总数据量: {len(df_all)}")
print(f"训练集 (75%): {len(train_df)} 条，保存在 {train_csv}")
print(f"测试集 (25%): {len(test_df)} 条，保存在 {test_csv}")