"""越線計數：以「移動線段是否與計數線相交」判斷物件是否真的跨線。

原專案只比較 y 座標是否小於某個固定值，物件只要出現在線的上方就會被計數，
也無法處理斜線或區分方向。這裡改成幾何上的線段相交判斷。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float]


def orient(a: Point, b: Point, c: Point) -> float:
    """點 c 相對於有向線段 a→b 的位置（外積）；正負號代表在哪一側，0 代表共線。"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """線段 p1p2 與 q1q2 是否嚴格相交（端點剛好落在線上不算）。"""
    d1 = orient(q1, q2, p1)
    d2 = orient(q1, q2, p2)
    d3 = orient(p1, p2, q1)
    d4 = orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


@dataclass
class LineCounter:
    """計數線，每個 track id 只計數一次。

    方向定義（影像座標，y 軸向下）：沿著 start→end 看，
    從外積為負的一側跨到正的一側算 "in"，反之算 "out"。
    例如 start 在左、end 在右的水平線，物件由上往下穿越就是 "in"。
    """

    start: Point
    end: Point
    count_in: int = 0
    count_out: int = 0
    crossed: dict[int, str] = field(default_factory=dict)
    _last: dict[int, Point] = field(default_factory=dict, repr=False)

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    def in_normal(self) -> Point | None:
        """垂直於計數線、指向 IN 一側的單位向量；起點和終點重疊時回傳 None。"""
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return None
        return (-dy / length, dx / length)

    def update(self, track_id: int, point: Point) -> str | None:
        """更新物件位置；若這一步跨越了計數線，回傳 "in" 或 "out"。"""
        # 剛好落在線上的點不當作「上一步位置」，避免跨線時被拆成兩段而漏算
        if orient(self.start, self.end, point) == 0:
            return None
        prev = self._last.get(track_id)
        self._last[track_id] = point
        if prev is None or track_id in self.crossed:
            return None
        if not segments_intersect(prev, point, self.start, self.end):
            return None

        direction = "in" if orient(self.start, self.end, prev) < 0 else "out"
        self.crossed[track_id] = direction
        if direction == "in":
            self.count_in += 1
        else:
            self.count_out += 1
        return direction
