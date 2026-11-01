"""用 Ultralytics 官方驗證器評估微調模型，輸出 Markdown 報告。

範例：
    python -m training.evaluate --weights weights/custom/visdrone-yolo26s.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_vision.models import load_model
from training.datasets import PROJECT_DIR, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="評估微調模型（Ultralytics model.val()）")
    parser.add_argument("--weights", required=True, help="微調後的權重")
    parser.add_argument("--data", default="visdrone", help="訓練時用的資料集")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", default=None, help="報告路徑，預設 reports/<權重名稱>-<split>.md")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    model = load_model(str(weights))
    m = model.val(data=str(prepare(args.data)), split=args.split, imgsz=args.imgsz,
                  plots=False, verbose=False)

    p, r, map50, map50_95 = (float(v) for v in m.mean_results())
    # 每張圖的毫秒數；postprocess 含 NMS（YOLO26 為端到端架構，幾乎不需要）
    speed = {k: round(float(v), 2) for k, v in m.speed.items() if k != "loss"}
    speed["total"] = round(sum(speed.values()), 2)
    classes = []
    for i, c in enumerate(m.ap_class_index):
        cp, cr, ap50, ap = (float(v) for v in m.class_result(i))
        classes.append({"class": m.names[c], "instances": int(m.nt_per_class[c]),
                        "precision": cp, "recall": cr, "ap50": ap50, "ap50_95": ap})

    lines = [
        f"# {weights.stem} 評估結果",
        "",
        f"- 資料集：{args.data}（{args.split}，{int(m.nt_per_class.sum())} 個標註）",
        f"- 輸入解析度：{args.imgsz}",
        "- 評估方式：Ultralytics `model.val()`",
        "",
        "| 指標 | 數值 |",
        "| --- | --- |",
        f"| Precision | {p:.3f} |",
        f"| Recall | {r:.3f} |",
        f"| mAP50 | {map50:.3f} |",
        f"| mAP50-95 | {map50_95:.3f} |",
        f"| 速度（ms / 張，前處理 + 推論 + 後處理） | {speed['total']:.2f} |",
        "",
        "| 類別 | 標註數 | Precision | Recall | AP50 | AP50-95 |",
        "| --- | --- | --- | --- | --- | --- |",
        *(f"| {c['class']} | {c['instances']} | {c['precision']:.3f} | {c['recall']:.3f} "
          f"| {c['ap50']:.3f} | {c['ap50_95']:.3f} |" for c in classes),
        "",
    ]

    default_out = PROJECT_DIR / "reports" / f"{weights.stem}-{args.split}.md"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    summary = {"precision": p, "recall": r, "map50": map50, "map50_95": map50_95}
    report = {"model": weights.stem, "data": args.data, "split": args.split, "imgsz": args.imgsz,
              "summary": summary, "speed_ms": speed, "classes": classes}
    out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
