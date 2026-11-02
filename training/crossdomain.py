"""跨資料集比較：COCO 預訓練與 VisDrone 微調模型，分別在 COCO 和 VisDrone 上評估。

兩個資料集的類別不同，只比共同的 6 類（person、bicycle、car、motorcycle、bus、truck）：
- VisDrone 標註換成 COCO 類別編號：行人 + 人（非行走）→ person、汽車 + 廂型車 → car，
  三輪車、有篷三輪車在 COCO 沒有對應，捨棄。
- VisDrone 微調模型的預測用同一張對照表轉成 COCO 編號。
- COCO 標註不必過濾：AP 是逐類別計算的，直接取這 6 類的 AP 再平均。

評估一律使用 Ultralytics 官方 model.val()。另外會用預訓練模型跑一次完整 80 類的 COCO 評估，
確認能重現官方公布的分數，證明評估流程本身沒有問題。

範例：
    python -m training.crossdomain
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ASSETS_URL, ROOT, YAML
from ultralytics.utils.downloads import download

from traffic_vision.models import load_model
from training.datasets import DATASETS_DIR, PROJECT_DIR, prepare

VISDRONE_TO_COCO = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}
COCO_NAMES: dict[int, str] = YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
COCO_IDS = {name: i for i, name in COCO_NAMES.items()}
SHARED = sorted({COCO_IDS[n] for n in VISDRONE_TO_COCO.values()})
ZH = {"person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "機車", "bus": "公車",
      "truck": "卡車"}

# (測試集, 模型, 評估解析度)：各資料集用自己的標準解析度，COCO 另外補一組 1024
EXPERIMENTS = [
    ("COCO", "pretrained", 640),
    ("COCO", "finetuned", 640),
    ("COCO", "finetuned", 1024),
    ("VisDrone", "pretrained", 1024),
    ("VisDrone", "finetuned", 1024),
]
KIND = {"pretrained": "COCO 預訓練", "finetuned": "VisDrone 微調"}


def coco_val() -> Path:
    """COCO val2017（5,000 張，約 1 GB）；不存在時下載，不下載 19 GB 的訓練集。"""
    root = DATASETS_DIR / "coco"
    if not (root / "images" / "val2017").exists():
        download([ASSETS_URL + "/coco2017labels.zip"], dir=DATASETS_DIR)
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=root / "images")
    data = DATASETS_DIR / "coco-val.yaml"
    YAML.save(data, {"path": str(root), "train": "images/val2017", "val": "images/val2017",
                     "names": COCO_NAMES})
    return data


def visdrone_as_coco(split: str) -> Path:
    """把 VisDrone 的 split 轉成 COCO 類別編號，回傳 data yaml。"""
    src_yaml = YAML.load(prepare("visdrone"))
    src = Path(src_yaml["path"])
    lut = {i: COCO_IDS.get(VISDRONE_TO_COCO.get(n, ""), -1) for i, n in src_yaml["names"].items()}
    out = DATASETS_DIR / "visdrone-coco"

    images = out / "images" / split
    if not images.exists():
        # 必須複製而不是建立符號連結：ultralytics 會解析連結，改讀原本的 VisDrone 標註
        shutil.copytree(src / "images" / split, images)
    labels = out / "labels" / split
    labels.mkdir(parents=True, exist_ok=True)
    for f in (src / "labels" / split).glob("*.txt"):
        rows = []
        for line in f.read_text().splitlines():
            cls, *xywh = line.split()
            if lut[int(cls)] >= 0:
                rows.append(" ".join([str(lut[int(cls)]), *xywh]))
        (labels / f.name).write_text("".join(r + "\n" for r in rows))
    labels.with_suffix(".cache").unlink(missing_ok=True)

    data = out / "data.yaml"
    YAML.save(data, {"path": str(out), "train": f"images/{split}", "val": f"images/{split}",
                     "names": COCO_NAMES})
    return data


def remap_validator(names: dict[int, str]) -> type[DetectionValidator]:
    """把 VisDrone 模型的預測類別轉成 COCO 編號，沒有對應的類別直接丟掉。"""
    lut = torch.tensor([COCO_IDS.get(VISDRONE_TO_COCO.get(names[i], ""), -1)
                        for i in range(len(names))])

    class RemapValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            # 預測已轉成 COCO 編號，類別數與名稱也要跟著換，否則 COCO 標註的編號會超出範圍
            self.names = self.metrics.names = COCO_NAMES
            self.nc = len(COCO_NAMES)

        def postprocess(self, preds):
            out = []
            for p in super().postprocess(preds):
                cls = lut.to(p["cls"].device)[p["cls"].long()]
                keep = cls >= 0
                out.append({"bboxes": p["bboxes"][keep], "conf": p["conf"][keep],
                            "cls": cls[keep].float(), "extra": p["extra"][keep]})
            return out

    return RemapValidator


def evaluate(weights: str, data: str, imgsz: int) -> dict:
    """在共同的 6 類上評估，回傳各類別與平均的 AP50、AP50-95。"""
    model = load_model(weights)
    if set(model.names.values()) == set(COCO_NAMES.values()):
        kwargs = {"classes": SHARED}
    else:
        kwargs = {"validator": remap_validator(model.names)}
    m = model.val(data=data, imgsz=imgsz, plots=False, verbose=False, **kwargs)
    row = {int(c): m.class_result(i) for i, c in enumerate(m.ap_class_index)}
    ap50 = {COCO_NAMES[c]: float(row[c][2]) if c in row else 0.0 for c in SHARED}
    ap = {COCO_NAMES[c]: float(row[c][3]) if c in row else 0.0 for c in SHARED}
    return {"ap50": ap50, "ap50_95": ap,
            "map50": sum(ap50.values()) / len(ap50), "map50_95": sum(ap.values()) / len(ap)}


def table(results: list[dict], metric: str, title: str) -> list[str]:
    classes = [COCO_NAMES[c] for c in SHARED]
    lines = [
        f"## {title}",
        "",
        "| 測試集 | 模型 | " + " | ".join(ZH[c] for c in classes)
        + " | **6 類平均** | 評估解析度 |",
        "| --- | --- |" + " --- |" * (len(classes) + 2),
    ]
    for r in results:
        cells = " | ".join(f"{r[metric][c]:.3f}" for c in classes)
        mean = r["map50" if metric == "ap50" else "map50_95"]
        lines.append(f"| {r['dataset']} | {KIND[r['model']]} | {cells} | **{mean:.3f}** "
                     f"| {r['imgsz']} |")
    return [*lines, ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO / VisDrone 跨資料集比較（共同 6 類）")
    parser.add_argument("--pretrained", default="yolo26s.pt")
    parser.add_argument("--finetuned", default="custom/visdrone-yolo26s.pt")
    parser.add_argument("--visdrone-split", default="test")
    parser.add_argument("--out", default=str(PROJECT_DIR / "reports" / "crossdomain.md"))
    args = parser.parse_args()

    data = {"COCO": str(coco_val()), "VisDrone": str(visdrone_as_coco(args.visdrone_split))}
    weights = {"pretrained": args.pretrained, "finetuned": args.finetuned}

    # 檢查評估流程：預訓練模型在完整 80 類上的分數應接近官方公布的值
    full = load_model(args.pretrained).val(data=data["COCO"], imgsz=640, plots=False,
                                           verbose=False)
    sanity = {"map50": float(full.box.map50), "map50_95": float(full.box.map)}

    results = []
    for dataset, model, imgsz in EXPERIMENTS:
        r = evaluate(weights[model], data[dataset], imgsz)
        results.append({"dataset": dataset, "model": model, "imgsz": imgsz, **r})
        print(f"{dataset} / {KIND[model]} / {imgsz}: mAP50 {r['map50']:.3f}, "
              f"mAP50-95 {r['map50_95']:.3f}")

    lines = [
        "# 跨資料集比較（YOLO26s，共同 6 類）",
        "",
        f"- COCO：val2017（5,000 張）；VisDrone：{args.visdrone_split}；Ultralytics `model.val()`",
        f"- 預訓練：`{args.pretrained}`（640 訓練）；"
        f"微調：`{args.finetuned}`（VisDrone，1024 訓練）",
        "- VisDrone 標註：行人 + 人（非行走）合併為人，汽車 + 廂型車合併為汽車，三輪車兩類不計",
        f"- 評估流程檢查：預訓練模型在 COCO 完整 80 類的 mAP50-95 為 {sanity['map50_95']:.3f}"
        f"（mAP50 {sanity['map50']:.3f}）",
        "",
        *table(results, "ap50", "mAP50"),
        *table(results, "ap50_95", "mAP50-95"),
    ]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    report = {"visdrone_split": args.visdrone_split, "pretrained": args.pretrained,
              "finetuned": args.finetuned, "coco_80_class": sanity, "results": results}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n".join(lines))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
