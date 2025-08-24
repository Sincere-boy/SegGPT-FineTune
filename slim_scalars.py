import csv

def downsample_every(rows, every):
    keep = []
    last_step = None
    for row in rows:
        step = int(float(row["step"]))
        if last_step is None or (step // every) > (last_step // every):
            keep.append(row)
            last_step = step
    return keep

infile = "scalars.csv"
outfile = "scalars_slim.csv"

special_tags = {"Training/Batch Loss", "LR Scheduler"}
every = 100

with open(infile, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 分组处理
by_tag = {}
for r in rows:
    by_tag.setdefault(r["tag"], []).append(r)

slimmed = []
for tag, lst in by_tag.items():
    if tag in special_tags:
        kept = downsample_every(lst, every)
    else:
        kept = lst
    slimmed.extend(kept)

# 排序 & 写回
slimmed.sort(key=lambda x: (x["tag"], float(x["step"])))

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value"])
    writer.writeheader()
    for r in slimmed:
        writer.writerow(r)

print(f"Done. wrote {len(slimmed)} rows to {outfile}")