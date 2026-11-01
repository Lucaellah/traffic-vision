"""在車輛資料集上微調 YOLO26，完成後把最佳權重放進 weights/custom/，App 會自動列出。

範例：
    python -m training.finetune                          # VisDrone + YOLO26s，100 epochs
    python -m training.finetune --epochs 3 --fraction 0.1 --name smoke   # 快速確認流程
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from traffic_vision.models import load_model, weights_path
from training.datasets import PROJECT_DIR, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="微調 YOLO26 車輛偵測模型")
    parser.add_argument("--data", default="visdrone", help="內建資料集簡稱或 YOLO 格式 yaml")
    parser.add_argument("--model", default="yolo26s.pt", help="起始權重（COCO 預訓練）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="空拍畫面的車很小，用比預設 640 更大的解析度")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="只用部分訓練資料")
    parser.add_argument("--patience", type=int, default=20, help="驗證集沒進步幾個 epoch 就停")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="例如 0 或 cpu；預設自動選擇")
    parser.add_argument("--name", default=None, help="輸出名稱，預設 <資料集>-<模型>")
    args = parser.parse_args()

    name = args.name or f"{Path(args.data).stem.lower()}-{Path(args.model).stem}"
    data_yaml = prepare(args.data)
    model = load_model(args.model)
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, patience=args.patience, workers=args.workers,
        device=args.device, project=str(PROJECT_DIR / "runs" / "finetune"), name=name,
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    dst = weights_path(f"custom/{name}.pt")
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dst)
    meta = {
        "name": name,
        "base_model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "fraction": args.fraction,
        "classes": model.names,
        "val_metrics": {k: round(float(v), 4) for k, v in results.results_dict.items()},
        "run_dir": str(best.parent.parent),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"已輸出 {dst}")


if __name__ == "__main__":
    main()
