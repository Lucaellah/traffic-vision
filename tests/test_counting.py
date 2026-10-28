from traffic_vision.counting import LineCounter, segments_intersect

H_LINE = ((0, 100), (200, 100))  # 左→右的水平線，y = 100


def test_segments_intersect():
    assert segments_intersect((50, 90), (50, 110), *H_LINE)
    assert not segments_intersect((50, 90), (50, 99), *H_LINE)
    # 線段延長線外的點不算
    assert not segments_intersect((250, 90), (250, 110), *H_LINE)


def test_top_to_bottom_is_in():
    c = LineCounter(*H_LINE)
    assert c.update(1, (50, 80)) is None
    assert c.update(1, (50, 120)) == "in"
    assert (c.count_in, c.count_out) == (1, 0)


def test_bottom_to_top_is_out():
    c = LineCounter(*H_LINE)
    c.update(1, (50, 120))
    assert c.update(1, (50, 80)) == "out"
    assert (c.count_in, c.count_out) == (0, 1)


def test_each_track_counted_once():
    c = LineCounter(*H_LINE)
    for y in (80, 120, 80, 120):
        c.update(1, (50, y))
    assert c.total == 1


def test_object_already_past_line_not_counted():
    # 原專案的 bug：只要 y < line 就計數，即使物件從沒跨線
    c = LineCounter(*H_LINE)
    for y in (50, 40, 30):
        c.update(1, (50, y))
    assert c.total == 0


def test_point_exactly_on_line_still_counts():
    c = LineCounter(*H_LINE)
    c.update(1, (50, 90))
    c.update(1, (50, 100))  # 剛好在線上
    assert c.update(1, (50, 110)) == "in"


def test_diagonal_line_and_multiple_tracks():
    c = LineCounter((0, 0), (100, 100))
    c.update(1, (60, 40))
    c.update(2, (40, 60))
    c.update(1, (40, 60))
    c.update(2, (60, 40))
    assert c.total == 2
    assert c.crossed[1] != c.crossed[2]
