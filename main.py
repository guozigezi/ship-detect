import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detect import *            # 你项目里的检测绘制函数
from tracking import Tracking   # 你项目里的跟踪逻辑

# --- 放在文件顶部 import 区域，且在 YOLO(args.model) 被调用之前 ---
# 1) PyTorch 2.6+ 的安全反序列化：补充白名单（可选，留作兜底）
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    add_safe_globals([DetectionModel, Sequential])
except Exception as e:
    print("Safe globals register failed:", e)

# 2) 更直接：临时“猴子补丁”成旧行为（仅在你信任权重时启用）
import torch as _torch
if not hasattr(_torch.load, "_patched_weights_only"):
    _orig_load = _torch.load

    def _unsafe_load(*args, **kwargs):
        # 关键：关闭安全精简加载，允许完整对象反序列化
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _unsafe_load._patched_weights_only = True
    _torch.load = _unsafe_load

# 设备选择
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


if __name__ == "__main__":
    """
    Args:
        imgsz (int): 输入图片尺寸，默认 640
        input (str): 单张图片推理输入路径，默认 337.png
        output (str): 单张图片推理输出名（保存到 ./Output/<output>.jpg）
        model (str): 模型路径，默认 ./Model/Boat-detect-medium.pt
        conf (float): 置信度阈值
        iou_threshold (float): IOU 阈值（用于 NMS/跟踪等）
        video (bool): 是否为视频输入
        detect (bool): 是否执行检测（而非训练/跟踪）
        tracking (bool): 是否在视频上做跟踪
        track_buffer (int): 跟踪丢失缓冲
        match_thresh (float): ByteTrack 匹配阈值
        time_check_state (float): 船舶状态更新时间
        train (bool): 是否训练
        epoch (int): 训练轮数
        batch (int): 训练 batch size（新增）
        workers (int): DataLoader workers（新增，Windows 建议 0~2）
    """
    parser = argparse.ArgumentParser(prog="Boat-detect", epilog="Ship detection/tracking & training")
    parser.add_argument("-imgsz", type=int, default=960, help="Input resolution for training/inference")
    parser.add_argument("-input", type=str, default="337.png", help="Path of input data")
    parser.add_argument("-output", type=str, default="output", help="Name of output image (saved to ./Output)")
    parser.add_argument("-model", type=str, default="./Model/Boat-detect-medium.pt", help="Path of model")
    parser.add_argument("-conf", type=float, default=0.6, help="Score confidence")
    parser.add_argument("-iou_threshold", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("-video", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Confirm input is a Video")
    parser.add_argument("-detect", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Activate task detection")
    parser.add_argument("-tracking", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Activate task tracking")
    parser.add_argument("-track_buffer", type=float, default=30, help="buffer to calculate the time when to remove tracks")
    parser.add_argument("-match_thresh", type=float, default=0.5, help="Matching threshold for tracking in ByteTrack")
    parser.add_argument("-time_check_state", type=float, default=1.5, help="Time to update state of ship")
    parser.add_argument("-train", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Task is training model")
    parser.add_argument("-epoch", type=int, default=50, help="Num epochs")

    # 新增：为 8GB 显卡友好，默认较小
    parser.add_argument("-batch", type=int, default=4, help="Train batch size (reduce if OOM)")
    parser.add_argument("-workers", type=int, default=2, help="DataLoader workers (0~2 on Windows)")
    parser.add_argument("-multiscale", action=argparse.BooleanOptionalAction, default=True, help="Enable multi-scale inference for small targets")
    parser.add_argument("--aux-imgsz", type=int, nargs="+", default=[640], dest="aux_imgsz", help="Additional inference resolutions when multiscale is enabled")

    args = parser.parse_args()

    # 清理一下显存碎片（如果是 CUDA）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 读取模型（此时 torch.load() 已被猴子补丁成 weights_only=False）
    model = YOLO(args.model)

    if args.train:
        # 训练：强制走 GPU（若可用），开启 AMP；把 batch/workers 透传进来，缓解 OOM
        model.train(
            data="data.yaml",
            epochs=args.epoch,
            imgsz=args.imgsz,
            single_cls=True,
            device="cuda" if torch.cuda.is_available() else device,
            batch=args.batch,
            workers=args.workers,
            amp=True,
            # 下面几个可按需解开以更省显存/更稳
            # freeze=10,          # 冻结前 10 层
            # mosaic=0.5,         # 降低马赛克增强强度
            # copy_paste=0.0,     # 关 Copy-Paste
        )
    else:
        # 推理 / 跟踪
        if args.video:
            if args.tracking:
                Tracking(args, model)
            else:
                detectVideo(args, model)
        else:
            # 单图检测
            img = cv2.imread(args.input)
            if img is None:
                raise FileNotFoundError(f"Input image not found: {args.input}")
            detect_img = Draw(
                model,
                img,
                conf=args.conf,
                iou_thresh=args.iou_threshold,
                imgsz=args.imgsz,
                multiscale=args.multiscale,
                aux_scales=args.aux_imgsz,
            )
            out_path = f"./Output/{args.output}.jpg"
            ok = cv2.imwrite(out_path, detect_img)
            if not ok:
                raise RuntimeError(f"Failed to write output image to {out_path}")
            print("The image was successfully detected")
            print(f"The image was successfully saved to {out_path}")
