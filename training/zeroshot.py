"""零樣本比較：COCO 預訓練模型和 VisDrone 微調模型，在兩者共有的類別上用官方 model.val() 評估。

COCO 和 VisDrone 的類別不同，所以先把 VisDrone 標註換成 COCO 類別編號：
行人 + 人（非行走）→ person、汽車 + 廂型車 → car；三輪車、有篷三輪車在 COCO 沒有對應，捨棄。
微調模型的預測也用同一張對照表轉成 COCO 編號，所有模型才能放在同一張表比較。

範例：
    python -m training.zeroshot yolo26s.pt custom/visdrone-yolo26s.pt --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ROOT, YAML

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


def build_dataset(split: str) -> Path:
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
                     split: f"images/{split}", "names": COCO_NAMES})
    return data


def remap_validator(names: dict[int, str]) -> type[DetectionValidator]:
    """把 VisDrone 模型的預測類別轉成 COCO 編號，沒有對應的類別直接丟掉。"""
    lut = torch.tensor([COCO_IDS.get(VISDRONE_TO_COCO.get(names[i], ""), -1)
                        for i in range(len(names))])

    class RemapValidator(DetectionValidator):
        def postprocess(self, preds):
            out = []
            for p in super().postprocess(preds):
                cls = lut.to(p["cls"].device)[p["cls"].long()]
                keep = cls >= 0
                out.append({"bboxes": p["bboxes"][keep], "conf": p["conf"][keep],
                            "cls": cls[keep].float(), "extra": p["extra"][keep]})
            return out

    return RemapValidator


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO 預訓練與 VisDrone 微調模型的共同類別比較")
    parser.add_argument("weights", nargs="+", help="權重檔，相對於 weights/ 或絕對路徑")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", default=None, help="預設 reports/zeroshot-<split>.md")
    args = parser.parse_args()

    data = str(build_dataset(args.split))
    results = []
    for w in args.weights:
        model = load_model(w)
        if set(model.names.values()) == set(COCO_NAMES.values()):
            kind, kwargs = "COCO 預訓練（零樣本）", {"classes": SHARED}
        else:
            kind, kwargs = "VisDrone 微調", {"validator": remap_validator(model.names)}
        m = model.val(data=data, split=args.split, imgsz=args.imgsz, plots=False, verbose=False,
                      **kwargs)
        p, r, map50, map50_95 = (float(v) for v in m.mean_results())
        per_class = {COCO_NAMES[int(c)]: float(m.class_result(i)[2])
                     for i, c in enumerate(m.ap_class_index)}
        results.append({"model": Path(w).stem, "kind": kind, "precision": p, "recall": r,
                        "map50": map50, "map50_95": map50_95, "ap50": per_class})

    classes = [COCO_NAMES[i] for i in SHARED]
    lines = [
        f"# 零樣本與微調比較（VisDrone {args.split}）",
        "",
        f"共同類別：{'、'.join(classes)}；輸入解析度 {args.imgsz}；Ultralytics `model.val()`。",
        "行人 + 人（非行走）合併為 person，汽車 + 廂型車合併為 car，三輪車兩類不計。",
        "",
        "| 模型 | 類型 | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | --- | --- | --- | --- | --- |",
        *(f"| {x['model']} | {x['kind']} | {x['precision']:.3f} | {x['recall']:.3f} "
          f"| {x['map50']:.3f} | {x['map50_95']:.3f} |" for x in results),
        "",
        "各類別 AP50：",
        "",
        "| 模型 | " + " | ".join(classes) + " |",
        "| --- |" + " --- |" * len(classes),
        *(f"| {x['model']} | " + " | ".join(f"{x['ap50'].get(c, 0.0):.3f}" for c in classes) + " |"
          for x in results),
        "",
    ]

    out = Path(args.out) if args.out else PROJECT_DIR / "reports" / f"zeroshot-{args.split}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    report = {"split": args.split, "imgsz": args.imgsz, "classes": classes, "results": results}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n".join(lines))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
