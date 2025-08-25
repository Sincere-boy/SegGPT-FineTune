#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import csv
from tensorboard.backend.event_processing import event_accumulator
import time

def find_event_files(path):
    """查找给定路径下的 TensorBoard 事件文件。"""
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))
    elif os.path.isfile(path):
        return [path]
    else:
        return []

def dump_events_to_rows(fp, prefix):
    """读取指定事件文件 fp，将其中的 scalar 数据整理为行字典列表。"""
    ea = event_accumulator.EventAccumulator(fp)
    t0 = time.time()
    print(f"[INFO]  - Reloading event file ...", flush=True)
    ea.Reload()
    dt = time.time() - t0
    tags = ea.Tags().get("scalars", [])
    print(f"[INFO]  - Reloaded. scalar tags: {len(tags)} (took {dt:.2f}s)", flush=True)
    run = prefix
    rows = []
    total = 0
    print(f"[INFO]  - Extracting scalars per tag ...", flush=True)
    for tag in tags:
        for ev in ea.Scalars(tag):
            rows.append({"run": run, "tag": tag, "step": ev.step, "value": ev.value})
        total += 1
        if total % 10 == 0:
            print(f"[INFO]    processed {total} tags ...", flush=True)
    return rows

def downsample_every(rows, every):
    """每隔 `every` 个 step 保留一条记录，用于降低数据采样密度。"""
    keep = []
    last_step = None
    for row in rows:
        step = int(float(row["step"]))
        if last_step is None or (step // every) > (last_step // every):
            keep.append(row)
            last_step = step
    return keep

def main():
    # 固定的事件文件路径，无需运行时传参
    event_path = "/root/autodl-tmp/logs/1756030541/events.out.tfevents.1756030541.autodl-container-10a44fbcf4-d104fbea.101302.0"
    print("[INFO] Starting to process event files...", flush=True)
    files = find_event_files(event_path)
    print(f"[INFO] Found {len(files)} event file(s)", flush=True)
    if not files:
        raise SystemExit(f"[ERR] 找不到事件文件: {event_path}")

    all_rows = []
    for fp in files:
        try:
            fsize = os.path.getsize(fp)
        except Exception:
            fsize = 0
        sz = fsize / (1024 * 1024)
        print(f"[INFO] Processing: {fp}  (~{sz:.1f} MB)", flush=True)
        run_prefix = os.path.dirname(fp) or "."
        all_rows += dump_events_to_rows(fp, run_prefix)

    print(f"[INFO] Writing {len(all_rows)} rows to scalars.csv", flush=True)
    # 将所有 scalar 数据写入 scalars.csv
    with open("scalars.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # 对特定 tag 下采样
    special_tags = {"Training/Batch Loss", "LR Scheduler"}
    every = 100
    by_tag = {}
    for r in all_rows:
        by_tag.setdefault(r["tag"], []).append(r)

    slimmed = []
    for tag, lst in by_tag.items():
        if tag in special_tags:
            kept = downsample_every(lst, every)
        else:
            kept = lst
        slimmed.extend(kept)

    # 排序并写出精简后的数据
    slimmed.sort(key=lambda x: (x["tag"], float(x["step"])))
    print(f"[INFO] Writing {len(slimmed)} rows to scalars_slim.csv", flush=True)
    with open("scalars_slim.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value"])
        writer.writeheader()
        for r in slimmed:
            writer.writerow(r)

    print(f"Done. wrote {len(all_rows)} rows to scalars.csv and {len(slimmed)} rows to scalars_slim.csv", flush=True)

if __name__ == "__main__":
    main()