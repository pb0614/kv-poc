import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

RAW, OUT = Path("data/raw"), Path("data/kv")
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
N_EVAL = 10

files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS)
if not files:
    sys.exit("data/raw 에 이미지가 없습니다.")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for i, src in enumerate(files, 1):
    kv_id = f"kv_{i:03d}"
    img = Image.open(src).convert("RGB")
    img.save(OUT / f"{kv_id}.png")  # 원본은 건드리지 않고 복사본 저장
    rows.append({"kv_id": kv_id, "file": f"data/kv/{kv_id}.png",
                 "source_name": src.name, "width": img.width, "height": img.height,
                 "product": "", "campaign": "", "split": "train"})

df = pd.DataFrame(rows)
random.seed(42)  # 매번 같은 평가셋이 뽑히도록 고정
eval_ids = random.sample(list(df.kv_id), min(N_EVAL, len(df)))
df.loc[df.kv_id.isin(eval_ids), "split"] = "eval"
df.to_csv("data/kv_list.csv", index=False, encoding="utf-8-sig")  # 엑셀에서 열어도 안 깨짐

print(f"{len(df)}장 정리 완료 (train {sum(df.split == 'train')}, eval {sum(df.split == 'eval')})")
print("\n해상도 분포:")
print(df.groupby(["width", "height"]).size().to_string())