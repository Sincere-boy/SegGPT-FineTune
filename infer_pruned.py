import os
import time
import numpy as np
from PIL import Image
import torch
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

# ===== 配置 =====
PRUNED_DIR = "result/test_dir/pruned_seggpt_40"   # 剪后模型保存目录（save_pretrained）
# INPUT_DIR  = "images/test_dir"
INPUT_DIR  = "/home/zjp/PycharmProjects/Painter/SegGPT/SegGPT_inference/data/ball"
OUTPUT_DIR = "/media/zjp/D/temp40"
# OUTPUT_DIR = "result/pruned_infer"
PROMPT_DIR = "prompt_images/ball"
FEATURE_ENSEMBLE = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 加载“裁剪后”的模型和处理器 =====
model = SegGptForImageSegmentation.from_pretrained(PRUNED_DIR, local_files_only=True).to(device).eval()
image_processor = SegGptImageProcessor.from_pretrained(PRUNED_DIR, local_files_only=True)

# ===== 读提示图/掩码 =====
prompt_image = Image.open(os.path.join(PROMPT_DIR, "save4_2.jpeg"))
prompt_mask  = Image.open(os.path.join(PROMPT_DIR, "save4_2_mask.jpeg")).convert("L")

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def overlay_mask_on_image(image, mask, color=(255,182,193), alpha=0.8):
    mask_bin = mask.convert("L").point(lambda p: 255 if p>0 else 0).convert("RGBA")
    arr = np.array(mask_bin)
    arr[arr[:,:,0] > 0] = [*color, int(255*alpha)]
    mask_rgba = Image.fromarray(arr)

    out = Image.new("RGBA", image.size, (255,255,255,0))
    out.paste(image.convert("RGBA"), (0,0))
    out.paste(mask_rgba, (0,0), mask_rgba)
    return out

# ===== 可选：预热（让 CUDA kernel 加载、cudnn 选择算法）=====
# 避免第一张图偏慢影响平均值
def warmup_once(img_path):
    img = Image.open(img_path)
    inputs = image_processor(
        images=img,
        prompt_images=[prompt_image],
        prompt_masks=[prompt_mask],
        return_tensors="pt",
        feature_ensemble=FEATURE_ENSEMBLE
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

# ===== 推理循环（计时） =====
paths = list_images(INPUT_DIR)
if not paths:
    raise FileNotFoundError(f"No images found in {INPUT_DIR}")

# 预热一次（用第一张图）
warmup_once(paths[0])

times = []
for p in paths:
    img = Image.open(p)
    inputs = image_processor(
        images=img,
        prompt_images=[prompt_image],
        prompt_masks=[prompt_mask],
        return_tensors="pt",
        feature_ensemble=FEATURE_ENSEMBLE
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ===== 只统计前向用时（精确的 GPU 计时需要同步）=====
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    times.append(elapsed_ms)
    print(f"[{os.path.basename(p)}] infer time: {elapsed_ms:.2f} ms")

    # ===== 后处理 & 保存可视化 =====
    target_sizes = [img.size[::-1]]   # (H, W)
    mask = image_processor.post_process_semantic_segmentation(outputs, target_sizes)[0]
    mask_img = Image.fromarray(mask.cpu().numpy().astype("uint8"))

    base = os.path.splitext(os.path.basename(p))[0]
    mask_img.save(os.path.join(OUTPUT_DIR, f"{base}_mask.png"))

    seg = np.array(img) * (np.array(mask_img)[:,:,None] > 0)
    Image.fromarray(seg.astype(np.uint8)).save(os.path.join(OUTPUT_DIR, f"{base}_seg.png"))

    overlay = overlay_mask_on_image(img, mask_img)
    overlay.save(os.path.join(OUTPUT_DIR, f"{base}_overlay.png"))

# ===== 汇总平均用时 =====
avg_ms = sum(times) / len(times)
print(f"\n[Summary] images: {len(times)}, avg infer time: {avg_ms:.2f} ms")