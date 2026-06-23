import os
import pandas as pd

base_dir = '/home/duomeitinrfx/data/Adience'
folds_dir = os.path.join(base_dir, 'folds')
faces_dir = os.path.join(base_dir, 'faces')

def process_txt_to_csv(txt_path, csv_path):
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path = parts[0]
                label = int(parts[1])
                # 拼接成 faces/ 下的绝对路径
                abs_path = os.path.join(faces_dir, rel_path)
                data.append({'image_path': abs_path, 'label': label})
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Generated CSV: {csv_path} with {len(df)} samples")

def main():
    for fold in range(5):
        fold_subfolder = os.path.join(folds_dir, f'test_fold_is_{fold}')
        txt_train = os.path.join(fold_subfolder, 'age_train.txt')
        txt_test = os.path.join(fold_subfolder, 'age_test.txt')
        
        csv_train = os.path.join(base_dir, f'Adience_train_fold{fold}.csv')
        csv_test = os.path.join(base_dir, f'Adience_test_fold{fold}.csv')
        
        if os.path.exists(txt_train):
            process_txt_to_csv(txt_train, csv_train)
        if os.path.exists(txt_test):
            process_txt_to_csv(txt_test, csv_test)
            
    # 将 fold 0 作为默认无泄露的 Adience_train.csv 和 Adience_test.csv 复制到根目录
    shutil_copy(os.path.join(base_dir, 'Adience_train_fold0.csv'), os.path.join(base_dir, 'Adience_train.csv'))
    shutil_copy(os.path.join(base_dir, 'Adience_test_fold0.csv'), os.path.join(base_dir, 'Adience_test.csv'))
    print("Copied fold 0 splits as default Adience_train.csv and Adience_test.csv")

def shutil_copy(src, dst):
    import shutil
    shutil.copy(src, dst)
    print(f"Copied {src} to {dst}")

if __name__ == '__main__':
    main()
