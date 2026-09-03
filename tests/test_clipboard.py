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


# --- what the user is told ----------------------------------------------------


class Model:
    def __init__(self, sees: bool) -> None:
        self.model = "qwen2.5-coder:14b"
        self._sees = sees

    def can_see(self) -> bool:
        return self._sees


def spoken(fn, *args) -> str:
    import io

    stream = io.StringIO()
    fn(*args, stream)
    return stream.getvalue()


def test_a_model_that_cannot_see_says_so_at_paste_time():
    # Said when the image is attached, not after the answer comes back: a model
    # that ignores a picture answers from the text, and that reads exactly like
    # an answer about the picture.
    from bkht.coder.cli import announce_image

    said = spoken(announce_image, Model(sees=False), "/tmp/one.png")
    assert "cannot see images" in said
    assert "/tmp/one.png" in said, "it must still say where the file went"
    assert "qwen2.5vl" in said, "and name a model that can"


def test_a_model_that_can_see_just_confirms():
    from bkht.coder.cli import announce_image

    said = spoken(announce_image, Model(sees=True), "/tmp/one.png")
    assert "cannot see" not in said
    assert "/tmp/one.png" in said


def test_a_provider_with_no_opinion_is_treated_as_blind():
    # `can_see` is newer than the provider interface, and a backend without it
    # must not have its silence read as yes.
    from bkht.coder.cli import announce_image

    said = spoken(announce_image, type("P", (), {"model": "old"})(), "/tmp/one.png")
    assert "cannot see images" in said


def test_an_empty_clipboard_says_so_rather_than_nothing(monkeypatch):
    # A keypress that appears to do nothing is worse than no keypress at all.
    from bkht.coder import cli

    monkeypatch.setattr(cli.clipboard, "helper_missing", lambda: "")
    monkeypatch.setattr(cli.clipboard, "read_image", lambda: None)
    said = spoken(lambda stream: cli.attach_image(stream))
    assert "no image on the clipboard" in said


def test_a_missing_helper_names_what_to_install(monkeypatch):
    from bkht.coder import cli

    monkeypatch.setattr(cli.clipboard, "helper_missing", lambda: "xclip")
    said = spoken(lambda stream: cli.attach_image(stream))
    assert "xclip" in said


def test_text_on_the_clipboard_is_not_saved_as_an_image(monkeypatch):
    from bkht.coder import cli

    monkeypatch.setattr(cli.clipboard, "helper_missing", lambda: "")
    monkeypatch.setattr(cli.clipboard, "read_image", lambda: b"just copied text")
    stream_said = spoken(lambda stream: cli.attach_image(stream))
    assert "no image on the clipboard" in stream_said


def test_an_image_is_encoded_once_and_then_remembered(tmp_path, monkeypatch):
    """The provider encodes every image on every round trip; a turn makes many."""
    monkeypatch.setattr(clipboard, "_ENCODED", {})
    path = tmp_path / "shot.png"
    path.write_bytes(b"first")

    reads = []
    real = type(path).read_bytes
    monkeypatch.setattr(
        type(path), "read_bytes", lambda self: (reads.append(self), real(self))[1]
    )

    assert clipboard.encode(path) == clipboard.encode(path) == clipboard.encode(path)
    assert len(reads) == 1


def test_an_image_replaced_at_the_same_path_is_encoded_again(tmp_path, monkeypatch):
    """A stale image is a wrong answer, not a slow one, so identity is checked."""
    monkeypatch.setattr(clipboard, "_ENCODED", {})
    path = tmp_path / "shot.png"

    path.write_bytes(b"first")
    before = clipboard.encode(path)
    path.write_bytes(b"second image entirely")
    assert clipboard.encode(path) != before


def test_a_deleted_image_still_raises_so_the_caller_can_drop_it(tmp_path, monkeypatch):
    """`_with_images` drops an unreadable path rather than losing the turn."""
    monkeypatch.setattr(clipboard, "_ENCODED", {})
    path = tmp_path / "gone.png"
    path.write_bytes(b"here")
    clipboard.encode(path)
    path.unlink()

    with pytest.raises(OSError):
        clipboard.encode(path)
    sent = _with_images({"role": "user", "content": "hi", "images": [str(path)]})
    assert "images" not in sent and sent["content"] == "hi"


def test_the_cache_does_not_grow_without_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(clipboard, "_ENCODED", {})
    for index in range(clipboard._ENCODED_LIMIT * 2):
        path = tmp_path / f"{index}.png"
        path.write_bytes(bytes([index]))
        clipboard.encode(path)
    assert len(clipboard._ENCODED) == clipboard._ENCODED_LIMIT
