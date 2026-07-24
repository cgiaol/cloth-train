import os
import argparse
from ultralytics import YOLO

def detect_drive_path():
    """自动感知当前平台，找到正确的 Google Drive 保存路径"""
    # 1. 如果是 Colab 挂载点
    if os.path.exists("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/yolo_clothing_exp"
    
    # 2. 如果是 Kaggle 挂载点 (假设你在 Kaggle 侧关联了 Google Drive 或使用默认路径)
    elif os.path.exists("/kaggle/working/google_drive"):
        return "/kaggle/working/google_drive/yolo_clothing_exp"
    elif os.path.exists("/kaggle/working"):
        # 如果 Kaggle 没挂载 Drive，降级保存在 Kaggle 本地 Working 目录
        return "/kaggle/working/yolo_clothing_exp"
    
    # 3. 本地或其他环境默认路径
    return "./yolo_clothing_exp"

def get_default_data_config():
    """根据环境自动寻找 data.yaml"""
    if os.path.exists("/kaggle/working/dataset/data.yaml"):
        return "/kaggle/working/dataset/data.yaml"
    elif os.path.exists("/content/dataset_yolo/data.yaml"):
        return "/content/dataset_yolo/data.yaml"
    return "configs/data.yaml"

def parse_args():
    default_save_dir = detect_drive_path()
    default_data_config = get_default_data_config()

    parser = argparse.ArgumentParser(description="YOLO 服装 9 细分类训练脚本 (Colab/Kaggle 双平台通用)")
    
    # 核心路径配置
    parser.add_argument("--data_config", type=str, default=default_data_config, help="data.yaml 路径")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="权重保存路径 (优先 Drive)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练基础模型")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50, help="总训练轮数")
    # Kaggle/Colab 通用推荐安全的 batch 与 workers
    parser.add_argument("--batch", type=int, default=32, help="Batch size (建议 16 或 32 规避 OOM)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像分辨率")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数 (Kaggle 建议 <= 4)")
    
    # 使用 action Stores 避免 argparse 的 bool 转换 Bug
    parser.add_argument("--cache", action="store_true", help="是否将图像缓存至 RAM")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 定义本次实验的完整输出路径与 last.pt 路径
    exp_name = "garment_9cls_run"
    last_ckpt = os.path.join(args.save_dir, exp_name, "weights", "last.pt")

    print(f"📁 权重保存根路径: {args.save_dir}")
    print(f"📄 使用数据配置文件: {args.data_config}")

    # 3. 核心：断点自动恢复 (Resume) 逻辑
    if os.path.exists(last_ckpt):
        print(f"\n🔄 发现历史权重文件: {last_ckpt}")
        print("✅ 正在自动启动 Resume 模式，恢复上次中断的训练进度...\n")
        
        # 直接加载中断时保存的 last.pt 权重
        model = YOLO(last_ckpt)
        # resume=True 会自动读取上次训练的参数和 Epoch 继续往下跑
        model.train(resume=True)
        
    else:
        print(f"\n🚀 未找到历史权重，加载预训练模型 [{args.model}] 开始全新的训练...")
        
        model = YOLO(args.model)
        model.train(
            data=args.data_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=args.cache,
            project=args.save_dir,     # 产出目录
            name=exp_name,              # 实验文件夹名称
            exist_ok=True,              # 允许在同一个文件夹内更新/覆盖
            save=True,                  # 保存 Checkpoint
            save_period=1               # 每个 Epoch 都保存一次，防止突然中断
        )

if __name__ == "__main__":
    main()
