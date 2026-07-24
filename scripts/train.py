import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 服装 9 细分类训练与断点续训脚本 (T4 极限优化版)")
    
    # 核心路径配置
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="data.yaml 的相对路径")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/yolo_clothing_exp", help="权重保存在 Google Drive 的路径")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练基础模型 (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    
    # 训练超参数 (已匹配 T4 极限参数)
    parser.add_argument("--epochs", type=int, default=50, help="总训练轮数")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (T4 GPU 极限建议 64)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像分辨率")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数 (Colab 推荐 8)")
    parser.add_argument("--cache", type=bool, default=True, help="是否将图像缓存至 RAM 以极速读取")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 确保 Google Drive 里的保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 定义本次实验在 Drive 中的完整输出路径与 last.pt 路径
    exp_name = "garment_9cls_run"
    last_ckpt = os.path.join(args.save_dir, exp_name, "weights", "last.pt")

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
        print(f"📊 使用配置文件: {args.data_config}\n")
        
        model = YOLO(args.model)
        model.train(
            data=args.data_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=args.save_dir,     # 将产出直接存入 Google Drive
            name=exp_name,              # 实验文件夹名称
            exist_ok=True,             # 允许在同一个文件夹内更新/覆盖
            save=True,                 # 保存 Checkpoint
            save_period=1              # 每个 Epoch 都保存一次，防止突然断网
        )

if __name__ == "__main__":
    main()
