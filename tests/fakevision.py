"""Offline stand-in for NVIDIA Nemotron Super VL in tests: it 'reads' a sample photo by
returning that sample's known transcription, so every test runs without a key or network."""

from pathlib import Path

from PIL import Image

from kept.capture import vision

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"


def _sig(p):
    return list(Image.open(p).convert("L").resize((16, 24)).getdata())


_SIGS = {p.stem: _sig(p) for p in (ROOT / "samples").glob("*.png")}


def which_sample(path):
    s = _sig(path)
    best = min(_SIGS, key=lambda k: sum(abs(a - b) for a, b in zip(_SIGS[k], s)))
    return best


def fake_read_photo(path, today, ai=None):
    name = which_sample(path)
    text = (FIX / f"{name}.txt").read_text()
    kind = "letter" if "letter" in name else "other"
    return {"text": text, "kind": kind, "ai": ai or {}, "handwritten_note": False}


def install(testcase_cls_or_none=None):
    from unittest import mock
    p = mock.patch.object(vision, "read_photo", side_effect=fake_read_photo)
    p.start()
    return p
