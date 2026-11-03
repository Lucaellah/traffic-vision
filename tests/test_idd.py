from training.idd import CLASS_ID, sequence, split_ids, yolo_lines

XML = """<annotation><object><name>motorcycle</name>
<bndbox><xmin>0</xmin><ymin>50</ymin><xmax>100</xmax><ymax>150</ymax></bndbox></object>
<object><name>caravan</name>
<bndbox><xmin>150</xmin><ymin>0</ymin><xmax>250</xmax><ymax>300</ymax></bndbox></object>
<object><name>train</name>
<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
</annotation>"""


def test_yolo_lines_merges_drops_and_clips():
    lines = [line.split() for line in yolo_lines(XML, 200, 200)]
    classes = [int(line[0]) for line in lines]
    assert classes == [CLASS_ID["motorcycle"], CLASS_ID["vehicle fallback"]]
    assert [float(v) for v in lines[0][1:]] == [0.25, 0.5, 0.5, 0.5]
    assert [float(v) for v in lines[1][1:]] == [0.875, 0.5, 0.25, 1.0]  # 超出圖片的部分裁掉


def test_sequence_joins_parts():
    assert sequence("hq/BLR-1_part_3/0001") == sequence("hq/BLR-1_part_7/0002") == "hq/BLR-1"


def test_split_keeps_sequences_apart():
    train = [f"a/s{s}/{f}" for s in range(20) for f in range(3)] + ["a/shared/0"]
    splits = split_ids(train, ["a/shared/1", "a/other/0"])
    seqs = {k: {sequence(i) for i in v} for k, v in splits.items()}
    assert not seqs["train"] & seqs["val"]
    assert not (seqs["train"] | seqs["val"]) & seqs["test"]  # 和測試集共用的序列不進訓練
    assert len(seqs["val"]) == 2
