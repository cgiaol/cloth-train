import os
import sys
import shutil
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from huggingface_hub import HfApi, hf_hub_download


# ==============================================================================
# 1. 细分类类别映射定义 (GarmentIQ / 服装 9 细分类)
# ==============================================================================
CATEGORY_MAP = {
    'dress': 'dress', 'dresses': 'dress',
    'top': 'top', 'tops': 'top', 'shirt': 'top', 't-shirt': 'top', 'blouse': 'top',
    'pant': 'pants', 'pants': 'pants', 'trousers': 'pants', 'jeans': 'pants',
    'skirt': 'skirt', 'skirts': 'skirt',
    'jacket': 'jacket', 'jackets': 'jacket', 'coat': 'jacket', 'outerwear': 'jacket',
    'sweater': 'sweater', 'sweaters': 'sweater', 'cardigan': 'sweater',
    'shorts': 'shorts', 'short': 'shorts',
    'suit': 'suit', 'suits': 'suit', 'blazer': 'suit',
    'shoe': 'shoes', 'shoes': 'shoes', 'footwear': 'shoes', 'sneaker': 'shoes', 'boot': 'shoes'
}

CLASSES = ['dress', 'top', 'pants', 'skirt', 'jacket', 'sweater', 'shorts', 'suit', 'shoes']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}


# ==============================================================================
# 2. 数据准备与清洗逻辑
# ==============================================================================
def map_category(cat_str):
    if not isinstance(cat_str, str):
        return None
    cat_str = cat_str.lower().strip()
    for key, mapped in CATEGORY_MAP.items():
        if key in cat_str:
            return mapped
    return None


def prepare_dataset(csv_path, dataset_dir, output_dir):
    """
    清洗 metadata.csv 并构建 YOLO 格式的分类数据集结构:
    output_dir/
      ├── images/
      │    ├── train/
      │    └── val/
      └── data.yaml
    """
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    
    # 如果数据集已经生成且 data.yaml 存在，跳过重新处理
    if os.path.exists(data_yaml_path) and os.path.exists(os.path.join(output_dir, "images", "train")):
        print(f"✅ 数据集已在 [{output_dir}] 准备就绪，跳过数据预处理。")
        return data_yaml_path

    print(f"📊 开始进行数据清洗与分类格式化...")
    print(f"  - 读取 CSV: {csv_path}")
    print(f"  - 数据源根目录: {dataset_dir}")
    print(f"  - 输出目录: {output_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 metadata.csv: {csv_path}")

    df = pd.read_csv(csv_path)

    category_col = next((c for c in ['category', 'label', 'class', 'garment_type'] if c in df.columns), None)
    img_col = next((c for c in ['image_path', 'image', 'file_name', 'path', 'filename'] if c in df.columns), None)

    if not category_col or not img_col:
        raise ValueError(f"无法自动识别分类列或图片路径列。CSV 包含列: {df.columns.tolist()}")

    df['mapped_category'] = df[category_col].apply(map_category)
    df_valid = df.dropna(subset=['mapped_category']).copy()
    
    print(f"  - 筛选有效分类样本数: {len(df_valid)} / {len(df)}")

    # 划分为训练集和验证集 (80% / 20%)
    train_df, val_df = train_test_split(
        df_valid, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_valid['mapped_category']
    )

    # 创建 YOLO 数据集目录架构
    for split in ['train', 'val']:
        for cls_name in CLASSES:
            os.makedirs(os.path.join(output_dir, "images", split, cls_name), exist_ok=True)

    def process_split(split_df, split_name):
        copied_count = 0
        for _, row in split_df.iterrows():
            rel_img_path = str(row[img_col]).lstrip('/')
            src_path = os.path.join(dataset_dir, rel_img_path)
            
            if not os.path.exists(src_path):
                alt_path = os.path.join(dataset_dir, os.path.basename(rel_img_path))
                if os.path.exists(alt_path):
                    src_path = alt_path
                else:
                    continue

            cls_name = row['mapped_category']
            dst_path = os.path.join(output_dir, "images", split_name, cls_name, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        return copied_count

    print("  - 正在复制训练集图片...")
    train_count = process_split(train_df, 'train')
    print("  - 正在复制验证集图片...")
    val_count = process_split(val_df, 'val')

    print(f"✅ 数据集构建完成！训练集: {train_count} 张，验证集: {val_count} 张。")

    # 写入 data.yaml 配置文件
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: dress
  1: top
  2: pants
  3: skirt
  4: jacket
  5: sweater
  6: shorts
  7: suit
  8: shoes
"""
    os.makedirs(output_dir, exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"📄 已自动生成配置文件: {data_yaml_path}")
    return data_yaml_path


# ==============================================================================
# 3. Hugging Face 云端同步逻辑
# ==============================================================================
def download_last_ckpt_from_hf(hf_repo, target_path):
    """尝试从 Hugging Face 私有/公开模型库拉取最新断点 last.pt"""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_repo or not hf_token:
        return False
        
    print(f"🔍 正在检查 Hugging Face [{hf_repo}] 是否存在上一次中断的权重文件 last.pt ...")
    try:
        downloaded_file = hf_hub_download(
            repo_id=hf_repo,
            filename="last.pt",
            repo_type="model",
            token=hf_token
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(downloaded_file, target_path)
        print(f"✅ 成功从 Hugging Face 拉取断点权重并存入本地: {target_path}")
        return True
    except Exception as e:
        print(f"💡 未在 Hugging Face 发现可用断点文件，将开启全新训练任务 ({e})")
        return False


def setup_hf_callback(model, hf_repo, hf_sync_period, save_dir, exp_name):
    """挂载 Ultralytics 训练回调函数：每 N 个 Epoch 自动静默上传至 Hugging Face"""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_repo or not hf_token:
        print("💡 [HF Sync] 未指定 --hf_repo 或未检测到环境变量 HF_TOKEN，跳过自动云端同步配置。")
        return

    api = HfApi()
    weights_dir = os.path.join(save_dir, exp_name, "weights")

    def on_train_epoch_end(trainer):
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        
        # 当到达同步周期倍数或是最后一轮时触发推流
        if current_epoch % hf_sync_period == 0 or current_epoch == total_epochs:
            print(f"\n🔔 [Epoch {current_epoch}/{total_epochs}] 触发每 {hf_sync_period} 轮自动推送至 Hugging Face...")
            for pt_name in ["last.pt", "best.pt"]:
                local_path = os.path.join(weights_dir, pt_name)
                if os.path.exists(local_path):
                    try:
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=pt_name,
                            repo_id=hf_repo,
                            token=hf_token,
                            repo_type="model"
                        )
                        print(f"  ✅ [HF Sync Successful] {pt_name} ➔ {hf_repo}/{pt_name}")
                    except Exception as e:
                        print(f"  ❌ [HF Sync Failed] {pt_name}: {e}")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    print(f"✅ 已成功挂载 Hugging Face 自动定时推流回调 (每 {hf_sync_period} 轮自动同步至 {hf_repo})")


# ==============================================================================
# 4. 主程序入口与命令参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 服装 9 细分类全流程一体化训练与自动续训脚本")
    
    # 数据集相关参数
    parser.add_argument("--csv_path", type=str, default="/kaggle/input/garmentiq-classification-set-nordstrom-and-myntra/metadata.csv", help="metadata.csv 的绝对/相对路径")
    parser.add_argument("--dataset_dir", type=str, default="/kaggle/input/garmentiq-classification-set-nordstrom-and-myntra", help="原始数据集图像解压根目录")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/dataset", help="格式化 YOLO 数据集输出路径")
    
    # 模型与训练参数
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="基础预训练模型 (yolov8n.pt / yolov8m.pt / yolov8x.pt)")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/yolo_clothing_exp", help="训练产出本地输出根目录")
    parser.add_argument("--epochs", type=int, default=50, help="总训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (T4 GPU 下 8x 推荐 16, 8n 推荐 64)")
    parser.add_argument("--imgsz", type=int, default=640, help="图像输入尺寸")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader 线程数")
    
    # Hugging Face 云端自动化参数
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face 目标 Repo 名称 (例如: username/yolo-clothing-weights)")
    parser.add_argument("--hf_sync_period", type=int, default=5, help="每隔多少个 Epoch 自动上传一次到 HF")

    return parser.parse_args()


def main():
    args = parse_args()

    print("======================================================================")
    print("🚀 启动 YOLOv8 服装细分类一站式训练系统 (Colab / Kaggle 通用版)")
    print("======================================================================\n")

    # Step 1: 自动预处理与清洗数据集
    data_yaml_path = prepare_dataset(
        csv_path=args.csv_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.data_dir
    )

    # Step 2: 准备训练输出路径
    os.makedirs(args.save_dir, exist_ok=True)
    exp_name = "garment_9cls_run"
    last_ckpt = os.path.join(args.save_dir, exp_name, "weights", "last.pt")

    # Step 3: 检查本地与 Hugging Face 权重断点
    if not os.path.exists(last_ckpt) and args.hf_repo:
        download_last_ckpt_from_hf(args.hf_repo, last_ckpt)

    # Step 4: 启动训练 (Resume 模式 或 全新开训)
    if os.path.exists(last_ckpt):
        print(f"\n🔄 发现断点权重: {last_ckpt}")
        print("✅ 正在启动 Resume 续训模式，自动接着上次轮次训练...\n")
        model = YOLO(last_ckpt)
        setup_hf_callback(model, args.hf_repo, args.hf_sync_period, args.save_dir, exp_name)
        model.train(resume=True)
    else:
        print(f"\n🚀 未找到历史断点，加载预训练模型 [{args.model}] 开始全新训练...")
        print(f"📊 使用数据配置: {data_yaml_path}\n")
        model = YOLO(args.model)
        setup_hf_callback(model, args.hf_repo, args.hf_sync_period, args.save_dir, exp_name)
        model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=True,
            amp=True,
            project=args.save_dir,
            name=exp_name,
            exist_ok=True,
            save=True,
            save_period=1
        )


if __name__ == "__main__":
    main()
