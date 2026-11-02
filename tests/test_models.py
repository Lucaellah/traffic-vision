import json

from traffic_vision import models


def test_custom_model_label_from_metadata(tmp_path, monkeypatch):
    custom = tmp_path / models.CUSTOM_DIR
    custom.mkdir()
    (custom / "visdrone-yolo26s.pt").touch()
    (custom / "visdrone-yolo26s.json").write_text(
        json.dumps({"base_model": "yolo26s.pt", "data": "visdrone"}))
    (custom / "mine.pt").touch()  # 沒有訓練紀錄：沿用檔名
    monkeypatch.setattr(models, "WEIGHTS_DIR", tmp_path)

    assert models.custom_models() == {
        "YOLO26s-visdrone（空拍 / 俯視）": "custom/visdrone-yolo26s.pt",
        "mine": "custom/mine.pt",
    }


def test_default_model_is_listed():
    assert models.DEFAULT_MODEL in models.available_models()
