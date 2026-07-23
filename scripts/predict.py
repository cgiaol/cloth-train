# Prediction/inference script for YOLO clothing detection
import os
import shutil
import pandas as pd

def convert_csv_to_yolo(csv_path, dataset_dir):
    df = pd.read_csv(csv_path)
    
    # 方案 B 的 9 细分类别精准映射表
    garment_map = {
        'long sleeve top': 0,
        'short sleeve top': 1,
        'vest': 2,
        'skirt': 3,
        'shorts': 4,
        'trousers': 5,
        'long sleeve dress': 6,
        'short sleeve dress': 7,
        'vest dress': 8
    }

    # 创建 YOLO 要求的目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    print(f"📊 正在处理 metadata.csv，共 {len(df)} 条记录...")

    missing_img_count = 0

    for idx, row in df.iterrows():
        filename = str(row['filename']).strip()
        garment_type = str(row['garment']).lower().strip()

        # 获取对应的类别 ID，如果没匹配上默认归为 1 (short sleeve top)
        cls_id = garment_map.get(garment_type, 1)

        # 8:2 划分训练集 (train) 与验证集 (val)
        split = 'val' if idx % 5 == 0 else 'train'

        # 1. 移动/归位图片文件到 YOLO 对应的 images/train 或 images/val
        src_img = os.path.join(dataset_dir, filename)
        dst_img = os.path.join(dataset_dir, 'images', split, filename)
        
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        else:
            # 兼容有些解压后图片在 images 文件夹里的情况
            src_img_alt = os.path.join(dataset_dir, 'images', filename)
            if os.path.exists(src_img_alt):
                shutil.move(src_img_alt, dst_img)
            else:
                missing_img_count += 1

        # 2. 生成对应的 YOLO 格式 txt 标注文件 (.txt)
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(dataset_dir, 'labels', split, txt_name)

        # 整张图片的全图标签标注（Center_X, Center_Y, Width, Height 分别为 0.5, 0.5, 1.0, 1.0）
        with open(txt_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    print(f"✅ 方案 B 数据转换完成！9 细分类标注已生成。")
    if missing_img_count > 0:
        print(f"⚠️ 提示：有 {missing_img_count} 张图片未在解压路径找到，已自动忽略。")

if __name__ == "__main__":
    convert_csv_to_yolo(
        csv_path="/content/dataset/metadata.csv",
        dataset_dir="/content/dataset"
    )
