# debug_val_pairs.py
import os, glob
from PIL import Image
import numpy as np

IMG_DIR = "data/val/images"
LBL_DIR = "data/val/labels"
EXPECT_RATIO = (2, 1)   # H:W 必须 2:1（你的 SegGPT 有这个硬约束）
PATCH = 16              # 尺寸最好还都是 16 的倍数
N_CLASSES = 8           # 按你的配置

def stem(p): 
    b = os.path.basename(p); return os.path.splitext(b)[0]

imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
lbls = sorted(glob.glob(os.path.join(LBL_DIR, "*")))
img_map = {stem(p): p for p in imgs}
lbl_map = {stem(p): p for p in lbls}

common = sorted(set(img_map) & set(lbl_map))
print(f"images={len(imgs)}, labels={len(lbls)}, common_pairs={len(common)}")
if not common:
    print("⚠️ 没有同名配对文件。")
    print("  images-only:", sorted(set(img_map)-set(lbl_map))[:5])
    print("  labels-only:", sorted(set(lbl_map)-set(img_map))[:5])

bad_ratio = []
bad_size_mismatch = []
bad_not_multiple_16 = []
bad_mask_empty = []
bad_mask_oor = []

for s in common:
    ip, lp = img_map[s], lbl_map[s]
    im = Image.open(ip).convert("RGB")
    lb = Image.open(lp)
    if lb.mode != "L":
        lb = lb.convert("L")

    iw, ih = im.size
    lw, lh = lb.size

    if (iw, ih) != (lw, lh):
        bad_size_mismatch.append((s, (iw, ih), (lw, lh)))
        continue

    # 比例 2:1
    if ih * EXPECT_RATIO[1] != iw * EXPECT_RATIO[0]:
        bad_ratio.append((s, (iw, ih)))

    # 16 的倍数
    if (iw % PATCH) or (ih % PATCH):
        bad_not_multiple_16.append((s, (iw, ih)))

    arr = np.array(lb, dtype=np.int32)
    uniq = np.unique(arr)
    if uniq.size <= 1:
        bad_mask_empty.append((s, uniq.tolist()))
    if uniq.max() >= N_CLASSES:
        bad_mask_oor.append((s, uniq.tolist()))

print("\n== 检查结果 ==")
print("尺寸不一致：", len(bad_size_mismatch))
print("比例非 2:1：", len(bad_ratio))
print("非 16 倍数：", len(bad_not_multiple_16))
print("掩码全背景/单值：", len(bad_mask_empty))
print("掩码含越界值(>=N_CLASSES)：", len(bad_mask_oor))

for title, items in [
    ("尺寸不一致样例", bad_size_mismatch),
    ("比例非2:1样例", bad_ratio),
    ("非16倍数样例", bad_not_multiple_16),
    ("掩码全背景样例", bad_mask_empty),
    ("掩码越界样例", bad_mask_oor),
]:
    if items:
        print(f"\n{title}（最多5个）：")
        for it in items[:5]:
            print("  ", it)
