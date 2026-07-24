# Prediction/inference script for YOLO clothing detection
import os
import shutil
import argparse
import pandas as pd

def get_default_paths():
    """根据运行环境自动识别默认路径 (Colab vs Kaggle)"""
    # 检查是否处于 Kaggle 环境
    if os.path.exists('/kaggle/input'):
        # 寻找 Kaggle 输入目录下的数据集文件夹
        input_dirs = [d for d in os.listdir('/kaggle/input') if os.path.isdir(os.path.join('/kaggle/input', d))]
        dataset_name = input_dirs[0] if input_dirs else 'clothing-dataset'
        
        base_input = os.path.join('/kaggle/input', dataset_name)
        return {
            'csv_path': os.path.join(base_input, 'metadata.csv'),
            'src_dataset_dir': base_input,
            'output_dir': '/kaggle/working/dataset'
        }
    else:
        # Colab / 本地 默认路径
        return {
            'csv_path': '/content/dataset/metadata.csv',
            'src_dataset_dir': '/content/dataset',
            'output_dir': '/content/dataset_yolo'
        }

def convert_csv_to_yolo(csv_path, src_dataset_dir, output_dir):
    print(f"🔍 启动数据预处理...")
    print(f"  - 读取 CSV: {csv_path}")
    print(f"  - 图片来源: {src_dataset_dir}")
    print(f"  - 输出目录: {output_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到 metadata.csv，请检查路径: {csv_path}")

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

    # 创建 YOLO 要求的目录结构 (输出到可写目录)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    print(f"📊 正在处理 metadata.csv，共 {len(df)} 条记录...")

    missing_img_count = 0

    for idx, row in df.iterrows():
        # 兼容列名，处理 filename 或 image 等列
        filename = str(row.get('filename', row.get('image', row.get('image_path', '')))).strip()
        garment_type = str(row.get('garment', row.get('category', ''))).lower().strip()

        if not filename:
            continue

        # 获取对应的类别 ID，如果没匹配上默认归为 1 (short sleeve top)
        cls_id = garment_map.get(garment_type, 1)

        # 8:2 划分训练集 (train) 与验证集 (val)
        split = 'val' if idx % 5 == 0 else 'train'

        # 1. 查找源图片路径 (兼容子目录和不同层级)
        src_img = os.path.join(src_dataset_dir, filename)
        if not os.path.exists(src_img):
            src_img = os.path.join(src_dataset_dir, 'images', filename)
            if not os.path.exists(src_img):
                src_img = os.path.join(src_dataset_dir, os.path.basename(filename))

        # 目标图片路径
        dst_img = os.path.join(output_dir, 'images', split, os.path.basename(filename))

        # 2. 复制图片 (Kaggle 环境输入层只读，必须使用 copy 而不能 move)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            missing_img_count += 1
            continue

        # 3. 生成对应的 YOLO 格式 txt 标注文件 (.txt)
        txt_name = os.path.splitext(os.path.basename(filename))[0] + ".txt"
        txt_path = os.path.join(output_dir, 'labels', split, txt_name)

        # 整张图片的全图标签标注（Center_X, Center_Y, Width, Height 分别为 0.5, 0.5, 1.0, 1.0）
        with open(txt_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    print(f"✅ 方案 B 数据转换完成！9 细分类标注已生成在 [{output_dir}]。")
    if missing_img_count > 0:
        print(f"⚠️ 提示：有 {missing_img_count} 张图片未找到源文件，已自动忽略。")

    # 4. 自动在输出目录下生成 data.yaml 配置文件
    data_yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: long sleeve top
  1: short sleeve top
  2: vest
  3: skirt
  4: shorts
  5: trousers
  6: long sleeve dress
  7: short sleeve dress
  8: vest dress
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    print(f"📄 已自动为你生成 YOLO 配置文件: {yaml_path}")

if __name__ == "__main__":
    defaults = get_default_paths()

    parser = argparse.ArgumentParser(description="YOLO 数据转换脚本 (Colab/Kaggle 双平台通用)")
    parser.add_argument("--csv_path", type=str, default=defaults['csv_path'], help="metadata.csv 路径")
    parser.add_argument("--src_dataset_dir", type=str, default=defaults['src_dataset_dir'], help="原始图片所在根目录")
    parser.add_argument("--output_dir", type=str, default=defaults['output_dir'], help="YOLO 格式数据集输出目录")

    args = parser.parse_args()

    convert_csv_to_yolo(
        csv_path=args.csv_path,
        src_dataset_dir=args.src_dataset_dir,
        output_dir=args.output_dir
    )
