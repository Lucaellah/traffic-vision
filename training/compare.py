"""把多份 evaluate.py 的 JSON 報告整理成一張比較表。

範例：
    python -m training.compare reports/visdrone-yolo26s-test.json reports/visdrone-yolo26m-test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.datasets import PROJECT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="比較多個模型的評估報告")
    parser.add_argument("reports", nargs="+", help="evaluate.py 輸出的 .json")
    parser.add_argument("--out", default=None, help="預設 reports/comparison-<split>.md")
    args = parser.parse_args()

    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    names = [r["model"] for r in reports]
    header = "| | " + " | ".join(names) + " |"
    sep = "| --- |" + " --- |" * len(names)

    def row(label: str, values: list[float], fmt: str = "{:.3f}", higher_better: bool = True):
        best = max(values) if higher_better else min(values)
        cells = [f"**{fmt.format(v)}**" if v == best and len(values) > 1 else fmt.format(v)
                 for v in values]
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "# 模型比較",
        "",
        f"資料集：{reports[0]['data']}（{reports[0]['split']}），輸入解析度 {reports[0]['imgsz']}，"
        "Ultralytics `model.val()`。粗體為最佳。",
        "",
        header, sep,
        row("mAP50", [r["summary"]["map50"] for r in reports]),
        row("mAP50-95", [r["summary"]["map50_95"] for r in reports]),
        row("Precision", [r["summary"]["precision"] for r in reports]),
        row("Recall", [r["summary"]["recall"] for r in reports]),
        row("速度（ms / 張）", [r["speed_ms"]["total"] for r in reports], "{:.2f}",
            higher_better=False),
        "",
        "各類別 AP50：",
        "",
        header, sep,
    ]
    per_class = [{c["class"]: c["ap50"] for c in r["classes"]} for r in reports]
    for cls in per_class[0]:
        lines.append(row(cls, [pc.get(cls, 0.0) for pc in per_class]))
    lines.append("")

    split = reports[0]["split"]
    out = Path(args.out) if args.out else PROJECT_DIR / "reports" / f"comparison-{split}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
