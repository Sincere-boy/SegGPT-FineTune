#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from tensorboard.backend.event_processing import event_accumulator

def find_event_files(path):
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))
    elif os.path.isfile(path):
        return [path]
    else:
        return []

def dump_file(fp, prefix=None):
    ea = event_accumulator.EventAccumulator(fp)
    ea.Reload()
    run = prefix if prefix is not None else os.path.dirname(fp) or "."

    # 只处理 scalars
    tags = ea.Tags().get("scalars", [])
    for tag in tags:
        for ev in ea.Scalars(tag):
            # ev: Event(wall_time, step, value)
            # 输出格式：run,tag,step,value
            print(f"{run},{tag},{ev.step},{ev.value}")

def main():
    ap = argparse.ArgumentParser(description="Dump TensorBoard scalar events to CSV (stdout).")
    ap.add_argument("path", help="事件文件或目录，如 logs/ 或 logs/1756021707/")
    ap.add_argument("--prefix", default=None,
                    help="当输入是单个文件时，指定输出中的 run 前缀；不指定则使用文件所在目录名")
    args = ap.parse_args()

    files = find_event_files(args.path)
    if not files:
        raise SystemExit(f"[ERR] 找不到事件文件: {args.path}")

    # 打印 CSV 头
    print("run,tag,step,value")
    for fp in files:
        run_prefix = args.prefix
        if run_prefix is None:
            # 用相对 logs 根的子路径做 run 名，尽量更可读
            run_prefix = os.path.relpath(os.path.dirname(fp), start=args.path) if os.path.isdir(args.path) else (os.path.dirname(fp) or ".")
        dump_file(fp, prefix=run_prefix)

if __name__ == "__main__":
    main()