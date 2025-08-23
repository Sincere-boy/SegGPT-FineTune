# python binarize_masks.py --one-value 255

import os
import argparse
from glob import glob
from PIL import Image
import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def list_images(d):
    return [p for ext in EXTS for p in glob(os.path.join(d, f"*{ext}"))]

def binarize_mask(path, threshold=0, one_value=1, dry_run=False):
    """
    将单通道/任意图像二值化为 0/1（或 0/255）。
    threshold: >threshold 记为 1，其余为 0
    one_value: 1 的写出数值（1 或 255）
    """
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    arr = np.array(im, dtype=np.int32)

    bin_arr = (arr > threshold).astype(np.uint8) * one_value  # 0 或 one_value
    if dry_run:
        return im.size, np.unique(arr).tolist(), np.unique(bin_arr).tolist()

    out = Image.fromarray(bin_arr.astype(np.uint8), mode="L")
    # 覆盖写回
    out.save(path)
    return im.size, None, None

def process_dir(d, threshold=0, one_value=1, dry_run=False):
    files = list_images(d)
    total = len(files)
    if total == 0:
        print(f"[WARN] 没在 {d} 找到图片")
        return
    changed = 0
    print(f"[INFO] 处理目录: {d}  共 {total} 张，阈值>{threshold} 记为 {one_value}")

    for i, p in enumerate(files, 1):
        try:
            size, orig_u, new_u = binarize_mask(p, threshold, one_value, dry_run)
            if dry_run:
                print(f"  [{i:>4}/{total}] {os.path.basename(p)}  size={size}  "
                      f"uniq_before={orig_u[:10]}...  uniq_after={new_u}")
            else:
                changed += 1
                if i % 50 == 0 or i == total:
                    print(f"  [{i:>4}/{total}] last={os.path.basename(p)}")
        except Exception as e:
            print(f"[ERROR] {p}: {repr(e)}")
    if not dry_run:
        print(f"[DONE] {d}: 成功写回 {changed}/{total} 张")

def main():
    ap = argparse.ArgumentParser(description="Binarize masks to 0/1 (grayscale).")
    ap.add_argument("--dirs", nargs="+", default=["data/train/labels", "data/val/labels"],
                    help="要处理的掩码目录（可多个）")
    ap.add_argument("--threshold", type=int, default=0,
                    help="像素值 > threshold => 1，否则 0（默认 0）")
    ap.add_argument("--one-value", type=int, default=1, choices=[1, 255],
                    help="输出中代表前景的数值（1 或 255，默认 1）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印统计信息，不写回文件")
    args = ap.parse_args()

    for d in args.dirs:
        process_dir(d, args.threshold, args.one_value, args.dry_run)

if __name__ == "__main__":
    main()
