"""訓練資料集的準備：下載、轉成 YOLO 格式，並產生指向專案內 datasets/ 的設定檔。"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"

# 簡稱 → ultralytics 內建的資料集設定
BUILTIN = {
    "visdrone": "VisDrone.yaml",
}


def prepare(data: str) -> Path:
    """回傳可以直接傳給 YOLO.train(data=...) 的 yaml 路徑；資料不存在時會自動下載。

    data 可以是內建簡稱（visdrone、idd），或是自己準備的 YOLO 格式 yaml。
    內建資料集會下載到專案的 datasets/，而不是 ultralytics 的全域資料夾。
    """
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    if data.lower() == "idd":  # 需要先自行下載（要註冊），見 training/idd.py
        from training.idd import convert

        out = DATASETS_DIR / "idd.yaml"
        return out if out.exists() else convert()
    if data.lower() not in BUILTIN:
        check_det_dataset(data)
        return Path(data)

    cfg = YAML.load(check_yaml(BUILTIN[data.lower()]))
    cfg["path"] = str(DATASETS_DIR / Path(cfg["path"]).name)
    cfg.pop("yaml_file", None)
    DATASETS_DIR.mkdir(exist_ok=True)
    out = DATASETS_DIR / f"{data.lower()}.yaml"
    YAML.save(out, cfg)
    check_det_dataset(str(out))  # 第一次會執行 yaml 裡的下載與轉換腳本
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下載並準備訓練資料集")
    parser.add_argument("data", nargs="?", default="visdrone")
    print(prepare(parser.parse_args().data))
