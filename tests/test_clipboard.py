"""Taking an image off the clipboard, and carrying it to the model."""

from __future__ import annotations

import struct
import zlib

import pytest

from bkht.coder import clipboard
from bkht.coder.provider import _with_images
from bkht.coder.session import Session


def png(width: int = 4, height: int = 4) -> bytes:
    """A real PNG, so the header check is exercised rather than mocked."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        payload = kind + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload))

    raw = b"".join(b"\x00" + bytes([255, 0, 0] * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_a_png_is_recognised():
    assert clipboard.looks_like_png(png()) is True


def test_other_bytes_are_not_a_png():
    # Text on the clipboard is the common case, and it must not be saved as an
    # image the model is then told to look at.
    assert clipboard.looks_like_png(b"just some copied text") is False
    assert clipboard.looks_like_png(b"") is False


def test_saving_writes_the_bytes_and_returns_the_path(tmp_path):
    path = clipboard.save(png(), directory=tmp_path)
    assert path.exists() and path.suffix == ".png"
    assert clipboard.looks_like_png(path.read_bytes())


def test_two_images_do_not_land_on_the_same_path(tmp_path):
    first = clipboard.save(png(4, 4), directory=tmp_path)
    second = clipboard.save(png(8, 8), directory=tmp_path)
    assert first != second


def test_encoding_round_trips(tmp_path):
    import base64

    path = clipboard.save(png(), directory=tmp_path)
    assert base64.b64decode(clipboard.encode(path)) == path.read_bytes()


def test_applescript_output_is_decoded():
    data = clipboard._from_applescript("«data PNGf" + png().hex() + "»")
    assert data == png()


def test_applescript_saying_nothing_is_not_an_image():
    assert clipboard._from_applescript("«class ficon»") is None


# --- carrying it to the model -------------------------------------------------


def test_the_session_keeps_paths_not_bytes(tmp_path):
    # A transcript is a file somebody may open; a megabyte of base64 in it
    # helps nobody, and the bytes are already on disk.
    path = str(clipboard.save(png(), directory=tmp_path))
    session = Session()
    session.add_user("what is this?", images=[path])
    assert session.messages[-1]["images"] == [path]


def test_a_message_with_no_images_is_untouched():
    message = {"role": "user", "content": "hello"}
    assert _with_images(message) == message


def test_paths_become_base64_on_the_way_out(tmp_path):
    path = str(clipboard.save(png(), directory=tmp_path))
    sent = _with_images({"role": "user", "content": "look", "images": [path]})
    assert sent["images"] == [clipboard.encode(path)]
    assert sent["content"] == "look"


def test_an_image_deleted_before_sending_costs_the_image_not_the_turn(tmp_path):
    path = clipboard.save(png(), directory=tmp_path)
    message = {"role": "user", "content": "look", "images": [str(path)]}
    path.unlink()
    sent = _with_images(message)
    assert "images" not in sent
    assert sent["content"] == "look"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_every_supported_platform_names_a_helper(platform, monkeypatch):
    # Either it works, or it says what to install. Silence is the one answer a
    # keypress that appears to do nothing must never give.
    monkeypatch.setattr(clipboard.sys, "platform", platform)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    assert clipboard.helper_missing() != ""
