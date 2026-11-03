"""IDD（India Driving Dataset）Detection 轉成 YOLO 格式。

IDD 是平視的行車畫面，機車、騎士、電動三輪車很多，接近台灣的車流。
官方 test 沒有公開標註，所以這裡的切法是：
- test：官方 val（訓練時完全不用，只拿來報告分數）
- val：從官方 train 依「影片序列」切出約 10%，用來挑最佳 epoch
- train：官方 train 的其餘部分

IDD 的圖是從連續影片取出的，相鄰幀幾乎一樣，所以 val 必須整段序列切出來，
不能隨機抽圖；官方 train 和 val 共用的少數序列也從 train 移除，避免測試集洩漏。

圖片用硬連結，不佔額外空間（符號連結會被 ultralytics 解析回原始路徑，找不到標註）。

範例：
    python -m training.idd            # 原始檔放在 datasets/IDD/IDD_Detection
"""

from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.datasets import DATASETS_DIR

SOURCE = DATASETS_DIR / "IDD" / "IDD_Detection"
OUT = DATASETS_DIR / "idd"

NAMES = [
    "person", "rider", "bicycle", "motorcycle", "car", "autorickshaw", "bus", "truck",
    "vehicle fallback", "animal", "traffic light", "traffic sign",
]
# 數量太少的類別：caravan、trailer 併入其他車輛；train（全部約 60 個框）捨棄
MERGE = {"caravan": "vehicle fallback", "trailer": "vehicle fallback"}
CLASS_ID = {n: i for i, n in enumerate(NAMES)}
VAL_FRACTION = 0.1


def sequence(image_id: str) -> str:
    """影像所屬的影片序列；highquality_16k 的 _part_N 是同一段影片切開的。"""
    return re.sub(r"_part_\d+$", "", image_id.rsplit("/", 1)[0])


def split_ids(train_ids: list[str], val_ids: list[str], seed: int = 0) -> dict[str, list[str]]:
    """依序列切出 train / val / test，確保同一段影片只出現在一個切分裡。"""
    test_seqs = {sequence(i) for i in val_ids}
    train_ids = [i for i in train_ids if sequence(i) not in test_seqs]
    seqs = sorted({sequence(i) for i in train_ids})
    random.Random(seed).shuffle(seqs)
    held = set(seqs[: max(1, round(len(seqs) * VAL_FRACTION))])
    return {
        "train": [i for i in train_ids if sequence(i) not in held],
        "val": [i for i in train_ids if sequence(i) in held],
        "test": list(val_ids),
    }


def yolo_lines(xml_text: str, width: int, height: int) -> list[str]:
    """把一張圖的 VOC XML 標註轉成 YOLO 格式（類別 中心x 中心y 寬 高，皆為 0～1）。"""
    lines = []
    for obj in ET.fromstring(xml_text).iter("object"):
        name = obj.findtext("name")
        name = MERGE.get(name, name)
        if name not in CLASS_ID:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
        y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{CLASS_ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return lines


def convert(source: Path = SOURCE, out: Path = OUT) -> Path:
    """產生 out/images、out/labels 和 datasets/idd.yaml，回傳 yaml 路徑。"""
    from PIL import Image
    from ultralytics.utils import TQDM, YAML

    ids = {s: (source / f"{s}.txt").read_text().split() for s in ("train", "val")}
    splits = split_ids(ids["train"], ids["val"])
    for split, image_ids in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for image_id in TQDM(image_ids, desc=f"轉換 {split}"):
            image = source / "JPEGImages" / f"{image_id}.jpg"
            stem = image_id.replace("/", "__")  # 各序列的檔名會重複，攤平成唯一名稱
            link = out / "images" / split / f"{stem}.jpg"
            if not link.exists():
                link.hardlink_to(image)
            with Image.open(image) as im:  # 以實際尺寸為準：少數 XML 寫 1921 寬
                width, height = im.size
            xml = (source / "Annotations" / f"{image_id}.xml").read_text(encoding="utf-8")
            lines = yolo_lines(xml, width, height)
            counts.update(int(line.split()[0]) for line in lines)
            (out / "labels" / split / f"{stem}.txt").write_text("".join(lines))
        print(f"{split}：{len(image_ids)} 張，{sum(counts.values())} 個框，"
              f"{len({sequence(i) for i in image_ids})} 段序列")

    yaml_path = DATASETS_DIR / "idd.yaml"
    YAML.save(yaml_path, {
        "path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(NAMES)),
    })
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把 IDD Detection 轉成 YOLO 格式")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(convert(args.source, args.out))
