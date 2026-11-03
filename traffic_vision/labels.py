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
    # VisDrone（微調模型）
    "pedestrian": "行人",
    "people": "人（非行走）",
    "van": "廂型車",
    "tricycle": "三輪車",
    "awning-tricycle": "有篷三輪車",
    "motor": "機車",
    # IDD（微調模型）
    "rider": "騎士",
    "autorickshaw": "電動三輪車",
    "vehicle fallback": "其他車輛",
    "animal": "動物",
    "traffic sign": "交通標誌",
}

# 統一的類別選單：不管選哪個模型，側邊欄都顯示同一組類別，再對應到模型自己的類別名稱
CATEGORIES = {
    "人": ["person", "pedestrian", "people", "rider"],
    "腳踏車": ["bicycle"],
    "機車": ["motorcycle", "motor"],
    "汽車": ["car", "van"],
    "公車": ["bus"],
    "卡車": ["truck"],
    "三輪車": ["tricycle", "awning-tricycle", "autorickshaw"],
    "其他車輛": ["vehicle fallback"],
    "紅綠燈": ["traffic light"],
}
# 影片頁預設勾選的類別
TRAFFIC_CATEGORIES = ["人", "腳踏車", "機車", "汽車", "公車", "卡車", "三輪車", "其他車輛"]


def category_ids(names: dict[int, str], categories: list[str]) -> list[int]:
    """把統一類別換成模型的類別編號；模型沒有的類別直接略過。"""
    wanted = {n for c in categories for n in CATEGORIES[c]}
    return sorted(i for i, n in names.items() if n in wanted)


def display_name(name: str) -> str:
    """回傳「中文 (english)」格式的顯示名稱。"""
    zh = ZH_NAMES.get(name)
    return f"{zh} ({name})" if zh else name
