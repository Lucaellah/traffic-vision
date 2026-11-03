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


def test_default_imgsz_follows_training_setting():
    class Fake:
        def __init__(self, overrides):
            self.overrides = overrides

    assert models.default_imgsz(Fake({"imgsz": 1024})) == 1024
    assert models.default_imgsz(Fake({})) == 640


def test_category_ids_map_unified_classes_per_model():
    from traffic_vision.labels import category_ids

    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane"}
    visdrone = {0: "pedestrian", 1: "people", 3: "car", 4: "van", 9: "motor"}
    assert category_ids(coco, ["人", "汽車"]) == [0, 2]
    assert category_ids(visdrone, ["人", "汽車", "機車"]) == [0, 1, 3, 4, 9]
    assert category_ids(coco, ["三輪車"]) == []  # 模型沒有的類別略過
