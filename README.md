# YOLO 服装检测系统（Clothing Detection with YOLO）

基于 YOLOv8 的服装类型检测系统，支持多种服装类别的实时检测。

## 📁 项目结构

```
yolo-clothing-detect/
├── configs/
│   └── data.yaml          # 核心！定义分类类别和训练/验证集路径
├── scripts/
│   ├── train.py           # 启动训练与Resume恢复脚本
│   └── predict.py         # 推理测试脚本
├── requirements.txt       # 项目依赖
└── README.md              # 项目文档
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/cgiaol/cloth-train.git
cd cloth-train

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将你的数据集按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

然后修改 `configs/data.yaml` 中的路径。

### 3. 模型训练

#### 首次训练

```bash
python scripts/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100 --batch 16
```

#### 从检查点恢复训练

```bash
python scripts/train.py --data configs/data.yaml --resume runs/detect/clothing_detection/weights/last.pt --epochs 150
```

#### 其他选项

```bash
python scripts/train.py --help
```

可用的模型大小：
- `yolov8n` - Nano（最小，速度最快）
- `yolov8s` - Small
- `yolov8m` - Medium
- `yolov8l` - Large
- `yolov8x` - Extra Large（精度最高，速度最慢）

### 4. 模型推理

#### 单张图片推理

```bash
python scripts/predict.py --model runs/detect/clothing_detection/weights/best.pt --source path/to/image.jpg --save
```

#### 视频推理

```bash
python scripts/predict.py --model runs/detect/clothing_detection/weights/best.pt --source path/to/video.mp4 --save
```

#### 实时摄像头推理

```bash
python scripts/predict.py --model runs/detect/clothing_detection/weights/best.pt --source 0
```

#### 推理选项

```bash
python scripts/predict.py --help
```

## 📊 配置文件说明

### configs/data.yaml

```yaml
path: /path/to/dataset      # 数据集根路径
train: images/train         # 训练集相对路径
val: images/val            # 验证集相对路径
test: images/test          # 测试集相对路径

nc: 10                     # 类别数量
names: [...]              # 类别名称列表
```

## 📈 训练结果

训练结果会保存在 `runs/detect/clothing_detection/` 目录中：

- `weights/best.pt` - 最优模型权重
- `weights/last.pt` - 最后一个检查点
- `results.csv` - 训练指标
- `plots/` - 可视化结果

## 🔧 关键特性

✅ 支持 YOLOv8 所有模型大小  
✅ 完整的训练恢复（Resume）功能  
✅ 灵活的推理选项（图片、视频、实时流）  
✅ 自动混合精度训练（AMP）  
✅ 早期停止（Early Stopping）  
✅ 易用的命令行接口  

## 📝 示例工作流

```bash
# 1. 环境设置
pip install -r requirements.txt

# 2. 训练（100个epoch）
python scripts/train.py --data configs/data.yaml --epochs 100 --batch 16

# 3. 模型评估
python scripts/predict.py --model runs/detect/clothing_detection/weights/best.pt --source dataset/images/test

# 4. 如果需要继续训练
python scripts/train.py --data configs/data.yaml --resume runs/detect/clothing_detection/weights/last.pt --epochs 150
```

## 💡 建议

- 在 GPU 上训练获得最佳性能
- 确保数据集标注格式正确（YOLO 格式）
- 根据硬件调整 `--batch` 大小
- 使用 `--patience` 参数启用早期停止

## 📚 参考资源

- [Ultralytics YOLO 文档](https://docs.ultralytics.com)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

## 📄 许可证

MIT License

---

**作者**: cgiaol  
**创建时间**: 2026-07-23
