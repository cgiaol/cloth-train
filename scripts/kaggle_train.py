import os
import argparse
import gdown

def parse_args():
    parser = argparse.ArgumentParser(description="从 Google Drive 下载训练断点 (last.pt) 以供 Kaggle/Colab 续训")
    
    # 接收 Google Drive 文件 ID 或分享链接
    parser.add_argument("--file_id", type=str, required=True, help="Google Drive 的文件 ID 或完整分享链接")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/yolo_clothing_exp", help="训练输出根目录")
    parser.add_argument("--exp_name", type=str, default="garment_9cls_run", help="实验文件夹名称")
    
    return parser.parse_args()

def extract_file_id(file_id_or_url):
    """从可能传入的完整 Google Drive URL 中自动提取纯 File ID"""
    if "drive.google.com" in file_id_or_url:
        if "/d/" in file_id_or_url:
            return file_id_or_url.split("/d/")[1].split("/")[0].split("?")[0]
        elif "id=" in file_id_or_url:
            return file_id_or_url.split("id=")[1].split("&")[0]
    return file_id_or_url.strip()

def main():
    args = parse_args()
    file_id = extract_file_id(args.file_id)
    
    # 拼接出 train.py 能识别的 weights/last.pt 路径
    target_dir = os.path.join(args.save_dir, args.exp_name, "weights")
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, "last.pt")
    
    print(f"📥 正在从 Google Drive 下载断点文件 (ID: {file_id})...")
    print(f"🎯 目标保存路径: {target_path}")
    
    # 执行下载
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, target_path, quiet=False)
    
    if os.path.exists(target_path):
        print(f"✅ 断点权重成功下载到指定目录！尺寸: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
        print("💡 接下来运行 scripts/train.py 将会自动检测并继续上次进度训练！")
    else:
        print("❌ 下载失败，请检查 Google Drive 文件的共享权限是否设置为『知道链接的任何人』！")

if __name__ == "__main__":
    main()
