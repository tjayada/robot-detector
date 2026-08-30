import json
import os

import pytest

from detect_utils import load_detections
from run_detect import (
    _append_journal,
    _close_journal,
    _open_journal,
    _write_json_atomic,
)


def test_journal_resume_round_trip(tmp_path):
    path = tmp_path / "detections.json.partial.jsonl"
    signature = {"dataset": "test", "frames": [0, 2]}
    entries = [
        {"image_id": 0, "found": True},
        {"image_id": 2, "found": False},
    ]

    journal, loaded = _open_journal(path, signature, resume=False)
    assert loaded == []
    for entry in entries:
        _append_journal(journal, entry, sync=False)
    _close_journal(journal)

    journal, loaded = _open_journal(path, signature, resume=True)
    _close_journal(journal)
    assert loaded == entries


def test_resume_rejects_changed_settings(tmp_path):
    path = tmp_path / "detections.json.partial.jsonl"
    journal, _ = _open_journal(path, {"segmentor": "sam"}, resume=False)
    _close_journal(journal)

    with pytest.raises(ValueError, match="settings do not match"):
        _open_journal(path, {"segmentor": "fastsam"}, resume=True)


def test_resume_discards_incomplete_final_record(tmp_path):
    path = tmp_path / "detections.json.partial.jsonl"
    signature = {"dataset": "test"}
    journal, _ = _open_journal(path, signature, resume=False)
    _append_journal(journal, {"image_id": 0, "found": True}, sync=True)
    _close_journal(journal)

    with open(path, "ab") as fh:
        fh.write(b'{"image_id":1')

    journal, loaded = _open_journal(path, signature, resume=True)
    _close_journal(journal)
    assert loaded == [{"image_id": 0, "found": True}]


def test_atomic_json_write(tmp_path):
    path = tmp_path / "detections.json"
    data = {"num_frames": 1, "detections": [{"image_id": 0}]}

    _write_json_atomic(path, data)

    assert json.loads(path.read_text()) == data
    assert not os.path.exists(f"{path}.tmp")


def test_gz_output_round_trips_through_the_shared_reader(tmp_path):
    # A .json.gz write must round-trip to the same document as a plain write.
    data = {"num_frames": 1, "detections": [{"image_id": 0, "frame_id": "000000"}]}
    plain, packed = tmp_path / "d.json", tmp_path / "d.json.gz"

    _write_json_atomic(plain, data)
    _write_json_atomic(packed, data)

    assert load_detections(packed) == load_detections(plain) == data
    assert packed.read_bytes()[:2] == b"\x1f\x8b"   # gzip magic
    assert not os.path.exists(f"{packed}.tmp")
