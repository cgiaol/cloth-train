import argparse
import os
import shutil
from ultralytics import YOLO


def detect_drive_path():
    """自动感知当前平台，找到正确的 Google Drive 保存路径"""
    # 1. 如果是 Colab 挂载点
    if os.path.exists("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/yolo_clothing_exp"

    # 2. 如果是 Kaggle 挂载点
    elif os.path.exists("/kaggle/working/google_drive"):
        return "/kaggle/working/google_drive/yolo_clothing_exp"
    elif os.path.exists("/kaggle/working"):
        # 降级保存在 Kaggle 本地 Working 目录
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

    parser = argparse.ArgumentParser(
        description="YOLO 服装 9 细分类训练脚本 (Colab/Kaggle 双平台通用)"
    )

    # 核心路径配置
    parser.add_argument(
        "--data_config",
        type=str,
        default=default_data_config,
        help="data.yaml 路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_save_dir,
        help="权重保存路径 (优先 Drive)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="预训练基础模型"
    )

    # 训练超参数
    parser.add_argument(
        "--epochs", type=int, default=50, help="总训练轮数"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (建议 16 或 32 规避 OOM)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="输入图像分辨率"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="数据加载线程数 (Kaggle 建议 <= 4)",
    )

    # 自动备份配置
    parser.add_argument(
        "--save_period", type=int, default=5, help="每几轮自动同步保存一次"
    )
    parser.add_argument(
        "--cache", action="store_true", help="是否将图像缓存至 RAM"
    )

    return parser.parse_args()


# 💡 核心：定义每 N 轮自动备份到 Drive/指定目录的回调函数
def setup_backup_callback(model, target_dir, period=5):
    def on_train_epoch_end(trainer):
        current_epoch = trainer.epoch + 1  # 当前跑完的轮数
        # 逢 5 的倍数轮或最后一轮触发备份
        if current_epoch % period == 0 or current_epoch == trainer.epochs:
            try:
                weights_dir = trainer.save_dir / "weights"
                last_pt = weights_dir / "last.pt"
                best_pt = weights_dir / "best.pt"

                backup_weights_dir = os.path.join(target_dir, "weights")
                os.makedirs(backup_weights_dir, exist_ok=True)

                if last_pt.exists():
                    shutil.copy(
                        last_pt, os.path.join(backup_weights_dir, "last.pt")
                    )
                    # 同时留一份按 Epoch 命名的备份
                    shutil.copy(
                        last_pt,
                        os.path.join(
                            backup_weights_dir, f"epoch_{current_epoch}.pt"
                        ),
                    )

                if best_pt.exists():
                    shutil.copy(
                        best_pt, os.path.join(backup_weights_dir, "best.pt")
                    )

                print(
                    f"\n☁️ [Backup] 第 {current_epoch} 轮训练完成，最新权重已成功同步备份至: {backup_weights_dir}"
                )
            except Exception as e:
                print(f"\n⚠️ [Backup Warning] 备份失败: {e}")

    # 注册回调事件
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


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

        model = YOLO(last_ckpt)
        setup_backup_callback(
            model,
            os.path.join(args.save_dir, exp_name),
            period=args.save_period,
        )
        model.train(resume=True)

    else:
        print(
            f"\n🚀 未找到历史权重，加载预训练模型 [{args.model}] 开始全新的训练..."
        )

        model = YOLO(args.model)
        setup_backup_callback(
            model,
            os.path.join(args.save_dir, exp_name),
            period=args.save_period,
        )
        model.train(
            data=args.data_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=args.cache,
            project=args.save_dir,
            name=exp_name,
            exist_ok=True,
            save=True,
            save_period=args.save_period,  # YOLO 本地每 period 轮存一个 checkpoint
        )


if __name__ == "__main__":
    main()
