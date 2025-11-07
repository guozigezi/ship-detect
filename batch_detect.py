
import os
import cv2
import torch
from ultralytics import YOLO
from main import Draw  # 使用项目里的 Draw 函数

# ========== 用户自定义部分 ==========
# 输入图片文件夹
input_dir = r"C:\ship_frames\北长沙"

# 输出文件夹
output_dir = r"C:\Users\Administrator\Desktop\Ship-detection-and-tracking-Yolov8-main1\ship_data\test_detected"

# 模型路径
model_path = r"C:\Users\Administrator\Desktop\Ship-detection-and-tracking-Yolov8-main1\Ship-detection-and-tracking-Yolov8-main1\runs\detect\train9\weights\best.pt"

# 要检测的最大图片数量
max_images = 200

# ========== 开始检测 ==========
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
model = YOLO(model_path)

# 支持的图片格式
img_exts = (".jpg", ".jpeg", ".png", ".bmp")

# 读取文件并限制数量
files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
files = sorted(files)[:max_images]  # 只取前100张

print(f"共检测到 {len(files)} 张图片（最多 {max_images} 张），开始检测...")

for i, filename in enumerate(files, start=1):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ 无法读取文件: {img_path}")
        continue

    # 调用 Draw 函数执行检测
    result_img = Draw(model, img)

    # 输出路径
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, result_img)
    print(f"[{i}/{len(files)}] 检测完成 -> {save_path}")

print("✅ 前 100 张图片检测完成！")
