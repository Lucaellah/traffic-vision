"""COCO 類別名稱的中文對照（只收錄交通場景常見類別，其餘沿用英文）。"""

ZH_NAMES = {
    "person": "行人",
    "bicycle": "腳踏車",
    "car": "汽車",
    "motorcycle": "機車",
    "bus": "公車",
    "train": "火車",
    "truck": "卡車",
    "traffic light": "紅綠燈",
    "fire hydrant": "消防栓",
    "stop sign": "停車標誌",
    "parking meter": "停車計費器",
    "bench": "長椅",
    "dog": "狗",
    "cat": "貓",
    # DOTA（空拍 OBB 模型）
    "small vehicle": "小型車",
    "large vehicle": "大型車",
    # VisDrone（微調模型）
    "pedestrian": "行人",
    "people": "人（非行走）",
    "van": "廂型車",
    "tricycle": "三輪車",
    "awning-tricycle": "有篷三輪車",
    "motor": "機車",
}

# 交通分析預設關注的類別（不存在於目前模型的會被略過）
TRAFFIC_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "small vehicle", "large vehicle",
    "pedestrian", "van", "tricycle", "awning-tricycle", "motor",
]


def display_name(name: str) -> str:
    """回傳「中文 (english)」格式的顯示名稱。"""
    zh = ZH_NAMES.get(name)
    return f"{zh} ({name})" if zh else name
