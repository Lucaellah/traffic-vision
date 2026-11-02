"""依 VisDrone 官方評估工具的規則評估微調模型，輸出 Markdown 報告。

Ultralytics 的 VisDrone 轉換腳本會直接刪掉忽略區域（類別 0）和 others（類別 11），
落在忽略區域裡的預測因此被算成誤報。官方工具（VisDrone2018-DET-toolkit）的規則：
- 標註物件和預測框只要有一半以上的面積落在忽略區域內，都不計分（dropObjectsInIgr.m）
- others 不屬於 10 個評估類別，預測也只有 10 類，所以等同不計
- 截斷程度不影響計分（標註只有 0%、1~50% 兩級）

官方的計分方式也和 Ultralytics 不同：每張圖最多取 500 個框，依信心分數由高到低貪婪配對，
以 VOC 方式計算 AP（calcAccuracy.m、evalRes.m）。這裡同一組預測算三種分數：
- Ultralytics 計分：和 training.evaluate 相同，作為對照
- Ultralytics 計分 + 忽略區域：只差在忽略區域，和專案其他數字可以直接比較
- 官方計分 + 忽略區域：完整依照官方工具，可以和論文、競賽的數字比較

範例：
    python -m training.visdrone_eval --weights weights/custom/visdrone-yolo26s.pt --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import YAML
from ultralytics.utils.metrics import ap_per_class, box_iou

from traffic_vision.models import load_model
from training.datasets import PROJECT_DIR, prepare

ZIPS = {"train": "train", "val": "val", "test": "test-dev"}
MAX_DETS = 500  # 官方工具每張圖最多取 500 個框
IGNORE_RATIO = 0.5  # 面積有一半以上落在忽略區域就不計分


def load_annotations(root: Path, split: str) -> dict[str, dict]:
    """從原始壓縮檔讀出 VisDrone 標註：10 類物件（編號從 0 開始）和忽略區域，座標為 xyxy 像素。"""
    out = {}
    with zipfile.ZipFile(root / f"VisDrone2019-DET-{ZIPS[split]}.zip") as z:
        for name in z.namelist():
            if "/annotations/" not in name or not name.endswith(".txt"):
                continue
            rows = [line.strip().rstrip(",").split(",")
                    for line in z.read(name).decode().strip().splitlines() if line.strip()]
            a = np.array([[float(v) for v in r[:6]] for r in rows]).reshape(-1, 6)
            xyxy = np.concatenate([a[:, :2], a[:, :2] + a[:, 2:4]], 1)
            obj = (a[:, 5] >= 1) & (a[:, 5] <= 10)
            out[Path(name).stem] = {"boxes": xyxy[obj], "cls": a[obj, 5] - 1,
                                    "ignored": a[a[:, 5] == 0, :4]}
    return out


def outside_ignored(boxes: np.ndarray, regions: np.ndarray, h: int, w: int) -> np.ndarray:
    """回傳哪些框不到一半面積落在忽略區域內（要保留計分的框），做法同 dropObjectsInIgr.m。"""
    if len(regions) == 0 or len(boxes) == 0:
        return np.ones(len(boxes), bool)
    mask = np.zeros((h + 1, w + 1))
    for x, y, rw, rh in np.maximum(regions, 0).astype(int):
        mask[y:min(h, y + rh) + 1, x:min(w, x + rw) + 1] = 1
    integral = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    b = np.round(boxes).astype(int)
    x0, y0 = np.clip(b[:, 0], 0, w), np.clip(b[:, 1], 0, h)
    x1, y1 = np.clip(b[:, 2], 0, w), np.clip(b[:, 3], 0, h)
    inside = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1)
    return inside / area < IGNORE_RATIO


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC 的 AP 算法（VOCap.m）：精確率取右側最大值後，對召回率的每段增量積分。"""
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))


def official_ap(images: list[dict], nc: int) -> np.ndarray:
    """官方計分（calcAccuracy.m + evalRes.m），回傳 (nc, 10) 的 AP；沒有標註的類別為 NaN。"""
    thresholds = np.linspace(0.5, 0.95, 10)
    ap = np.full((nc, 10), np.nan)
    for c in range(nc):
        n_gt = sum(int((im["gt_cls"] == c).sum()) for im in images)
        if n_gt == 0:
            continue
        scores, hits = [], [[] for _ in thresholds]
        for im in images:
            d = im["pred_cls"] == c  # 預測已依分數排序並截到 500 個
            g = im["gt_cls"] == c
            conf = im["pred_conf"][d]
            scores.append(conf)
            if not d.any():
                continue
            if not g.any():
                for h in hits:
                    h.append(np.zeros(len(conf), bool))
                continue
            iou = box_iou(torch.tensor(im["pred_boxes"][d]),
                          torch.tensor(im["gt_boxes"][g])).numpy()
            for t, thr in enumerate(thresholds):
                tp = np.zeros(len(conf), bool)
                free = np.ones(iou.shape[1], bool)
                for i in np.flatnonzero(iou.max(1) >= thr):  # 分數高的先配對
                    cand = np.where(free, iou[i], 0)
                    j = int(cand.argmax())
                    if cand[j] >= thr:
                        tp[i], free[j] = True, False
                hits[t].append(tp)
        conf = np.concatenate(scores)
        order = np.argsort(-conf, kind="stable")
        for t in range(len(thresholds)):
            tp = np.concatenate(hits[t])[order] if hits[t] else np.zeros(0, bool)
            ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
            ap[c, t] = voc_ap(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)) if len(tp) else 0.0
    return ap


def ultralytics_ap(images: list[dict], nc: int) -> np.ndarray:
    """Ultralytics 的配對與 AP 算法（和 model.val() 相同），回傳 (nc, 10) 的 AP。"""
    matcher = DetectionValidator()
    matcher.iouv = torch.linspace(0.5, 0.95, 10)
    tps = []
    for im in images:
        if len(im["gt_cls"]) and len(im["pred_cls"]):
            iou = box_iou(torch.tensor(im["gt_boxes"]), torch.tensor(im["pred_boxes"]))
            tps.append(np.asarray(matcher.match_predictions(
                torch.tensor(im["pred_cls"]), torch.tensor(im["gt_cls"]), iou)))
        else:
            tps.append(np.zeros((len(im["pred_cls"]), 10), bool))
    r = ap_per_class(np.concatenate(tps), np.concatenate([im["pred_conf"] for im in images]),
                     np.concatenate([im["pred_cls"] for im in images]),
                     np.concatenate([im["gt_cls"] for im in images]))
    ap = np.full((nc, 10), np.nan)
    ap[r[6]] = r[5]
    return ap


def predict(weights: Path, data: Path, split: str, imgsz: int, tmp: Path) -> dict[str, dict]:
    """用官方驗證器推論（和 model.val() 完全相同的前處理與 NMS），回傳每張圖的預測。"""
    load_model(str(weights)).val(data=str(data), split=split, imgsz=imgsz, max_det=MAX_DETS,
                                 save_json=True, plots=False, verbose=False,
                                 project=str(tmp), name="pred", exist_ok=True)
    preds = {}
    for p in json.loads((tmp / "pred" / "predictions.json").read_text()):
        preds.setdefault(p["image_id"], []).append(p)
    out = {}
    for stem, ps in preds.items():
        ps.sort(key=lambda p: -p["score"])
        b = np.array([p["bbox"] for p in ps[:MAX_DETS]])
        out[stem] = {"boxes": np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], 1),
                     "conf": np.array([p["score"] for p in ps[:MAX_DETS]]),
                     # save_json 的類別編號從 1 開始
                     "cls": np.array([p["category_id"] - 1 for p in ps[:MAX_DETS]], float)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="依 VisDrone 官方規則評估（忽略區域、官方計分）")
    parser.add_argument("--weights", required=True, help="微調後的權重")
    parser.add_argument("--split", default="test", choices=list(ZIPS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", default=None,
                        help="報告路徑，預設 reports/<權重名稱>-<split>-official.md")
    args = parser.parse_args()

    from PIL import Image

    weights = Path(args.weights).resolve()
    data = prepare("visdrone")
    cfg = YAML.load(data)
    root, names = Path(cfg["path"]), cfg["names"]
    nc = len(names)
    tmp = PROJECT_DIR / "runs" / "visdrone_eval"
    preds = predict(weights, data, args.split, args.imgsz, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    empty = {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0)}
    raw, kept = [], []
    dropped_gt = dropped_pred = 0
    for stem, ann in sorted(load_annotations(root, args.split).items()):
        p = preds.get(stem, empty)
        im = {"gt_boxes": ann["boxes"], "gt_cls": ann["cls"], "pred_boxes": p["boxes"],
              "pred_conf": p["conf"], "pred_cls": p["cls"]}
        raw.append(im)
        w, h = Image.open(root / "images" / args.split / f"{stem}.jpg").size
        g = outside_ignored(ann["boxes"], ann["ignored"], h, w)
        d = outside_ignored(p["boxes"], ann["ignored"], h, w)
        dropped_gt += int((~g).sum())
        dropped_pred += int((~d).sum())
        kept.append({"gt_boxes": ann["boxes"][g], "gt_cls": ann["cls"][g],
                     "pred_boxes": p["boxes"][d], "pred_conf": p["conf"][d],
                     "pred_cls": p["cls"][d]})

    scores = {"ultralytics": ultralytics_ap(raw, nc),
              "ultralytics_ignore": ultralytics_ap(kept, nc),
              "official_ignore": official_ap(kept, nc)}
    summary = {k: {"map50": float(np.nanmean(ap[:, 0])), "map50_95": float(np.nanmean(ap))}
               for k, ap in scores.items()}
    classes = [{"class": names[c], **{k: float(ap[c, 0]) for k, ap in scores.items()}}
               for c in range(nc)]
    n_gt = sum(len(im["gt_cls"]) for im in raw)
    n_pred = sum(len(im["pred_cls"]) for im in raw)

    titles = {"ultralytics": "Ultralytics 計分",
              "ultralytics_ignore": "Ultralytics 計分 + 忽略區域",
              "official_ignore": "官方計分 + 忽略區域"}
    lines = [
        f"# {weights.stem} 評估結果（VisDrone 官方規則）",
        "",
        f"- 資料集：VisDrone {ZIPS[args.split]}（{len(raw)} 張，{n_gt} 個標註）",
        f"- 輸入解析度：{args.imgsz}；每張圖最多 {MAX_DETS} 個框",
        f"- 忽略區域：面積一半以上落在其中的不計分，排除了 {dropped_gt} 個標註"
        f"（{dropped_gt / max(n_gt, 1):.1%}）、{dropped_pred} 個預測"
        f"（{dropped_pred / max(n_pred, 1):.1%}）",
        "- 官方計分：依 VisDrone2018-DET-toolkit（信心分數高的先配對、VOC 方式計算 AP）",
        "",
        "| 計分方式 | mAP50 | mAP50-95 |",
        "| --- | --- | --- |",
        *(f"| {titles[k]} | {s['map50']:.3f} | {s['map50_95']:.3f} |" for k, s in summary.items()),
        "",
        "各類別 AP50：",
        "",
        "| 類別 | " + " | ".join(titles.values()) + " |",
        "| --- |" + " --- |" * len(titles),
        *(f"| {c['class']} | " + " | ".join(f"{c[k]:.3f}" for k in titles) + " |"
          for c in classes),
        "",
    ]

    default_out = PROJECT_DIR / "reports" / f"{weights.stem}-{args.split}-official.md"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    report = {"model": weights.stem, "split": args.split, "imgsz": args.imgsz,
              "max_dets": MAX_DETS, "dropped_gt": dropped_gt, "dropped_pred": dropped_pred,
              "summary": summary, "classes": classes}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n".join(lines))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
