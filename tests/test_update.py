"""The release check, and `coder update`.

Nothing here touches the network: the `no_update_check` fixture in conftest
already points the cache at a temp file and stubs `refresh`, and the tests that
care about a request install their own fake in place of it.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from bkht.coder import update

#: Captured at import, before the autouse fixture stubs it out. The tests below
#: that exercise the request itself need the real thing.
REFRESH = update.refresh


@pytest.fixture
def installed(monkeypatch):
    """A copy on PATH, rather than a checkout: the case a notice is for."""
    monkeypatch.setattr(update, "version", lambda: "0.2.0")
    monkeypatch.setattr(update, "editable", lambda: None)
    monkeypatch.setattr(update.terminal, "interactive", lambda *a, **k: True)


def cache(stored: dict) -> None:
    update.CACHE.write_text(json.dumps(stored), encoding="utf-8")


def responder(payload, status: int = 200):
    """A stand-in for `httpx.get` that returns one canned response."""

    def get(url, **kwargs):
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))

    return get


# ------------------------------------------------------------------ ordering


@pytest.mark.parametrize("text", ["0.3.0", "v0.3.0", "V0.3.0", "1.2", "0.3.0.dev4+g9ab24a6"])
def test_a_version_that_names_a_release_parses(text):
    assert update.parse(text) is not None


@pytest.mark.parametrize("text", ["", "   ", "latest", "main", "vNext", None])
def test_anything_that_does_not_name_a_release_does_not(text):
    assert update.parse(text) is None


def test_releases_order_by_number_not_by_string():
    # The string comparison this replaces reads 0.10.0 as older than 0.9.0.
    assert update.newer("0.10.0", "0.9.0")
    assert update.newer("1.0.0", "0.99.99")
    assert not update.newer("0.9.0", "0.10.0")


def test_a_v_prefix_is_not_part_of_the_version():
    assert not update.newer("v0.3.0", "0.3.0")
    assert not update.newer("0.3.0", "v0.3.0")


def test_a_trailing_zero_is_not_a_newer_release():
    assert not update.newer("0.3", "0.3.0")
    assert not update.newer("0.3.0", "0.3")


def test_a_dev_version_is_below_the_release_it_leads_to():
    # The case that matters: a checkout heading for 0.3.0 must be told that
    # 0.3.0 exists, and must not be told that about the version it already has.
    assert update.newer("0.3.0", "0.3.0.dev4+g9ab24a6")
    assert not update.newer("0.3.0.dev4+g9ab24a6", "0.3.0")


def test_a_version_that_cannot_be_read_is_never_newer():
    # "Cannot tell" has to be silent. Announcing an update on an unparseable
    # string would send somebody to reinstall over a version that is fine.
    assert not update.newer("latest", "0.2.0")
    assert not update.newer("0.3.0", "")


# --------------------------------------------------------------------- cache


def test_a_fresh_cache_is_not_asked_about_again():
    cache({"checked": time.time(), "latest": "0.3.0"})
    assert not update.stale()


def test_a_day_old_cache_is():
    cache({"checked": time.time() - update.INTERVAL - 1, "latest": "0.3.0"})
    assert update.stale()


def test_no_cache_at_all_is_stale():
    assert update.stale()


def test_a_corrupt_cache_is_stale_rather_than_fatal():
    update.CACHE.write_text("{not json", encoding="utf-8")
    assert update.stale()
    assert update.cached() is None


def test_a_successful_check_caches_the_release(monkeypatch):
    monkeypatch.setattr(update.httpx, "get", responder({"tag_name": "v0.4.0"}))
    REFRESH()
    assert update.cached() == "0.4.0"


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("too slow"),
])
def test_a_failed_check_is_swallowed_but_still_stamped(monkeypatch, failure):
    # Stamped even on failure: without it a machine with no network would ask
    # again on every single launch, which is the one way a check this
    # unimportant becomes expensive.
    def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(update.httpx, "get", refuse)
    REFRESH()

    assert update.cached() is None
    assert not update.stale()


def test_a_rate_limited_check_is_swallowed(monkeypatch):
    monkeypatch.setattr(update.httpx, "get", responder({"message": "rate limited"}, status=403))
    REFRESH()
    assert update.cached() is None
    assert not update.stale()


@pytest.mark.parametrize("payload", [{}, {"tag_name": None}, {"tag_name": "nightly"}, []])
def test_a_shape_the_api_has_never_returned_is_swallowed(monkeypatch, payload):
    monkeypatch.setattr(update.httpx, "get", responder(payload))
    REFRESH()
    assert update.cached() is None
    assert not update.stale()


def test_a_failed_check_keeps_the_release_it_already_knew(monkeypatch):
    cache({"checked": 0, "latest": "0.3.0"})

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(update.httpx, "get", refuse)
    REFRESH()
    assert update.cached() == "0.3.0"


# -------------------------------------------------------------------- notice


def test_a_newer_release_is_announced(installed):
    cache({"checked": time.time(), "latest": "0.3.0"})
    assert "0.3.0" in update.notice()
    assert "coder update" in update.notice()


def test_the_release_you_are_on_is_not(installed):
    cache({"checked": time.time(), "latest": "0.2.0"})
    assert update.notice() == ""


def test_nothing_is_announced_off_a_terminal(installed, monkeypatch):
    # A piped run must not phone home, and nobody would read the line anyway.
    cache({"checked": time.time(), "latest": "0.3.0"})
    monkeypatch.setattr(update.terminal, "interactive", lambda *a, **k: False)
    assert update.notice() == ""


def test_nothing_is_announced_when_the_setting_is_off(installed):
    cache({"checked": time.time(), "latest": "0.3.0"})
    assert update.notice(SimpleNamespace(update_check=False)) == ""
    assert update.notice(SimpleNamespace(update_check=True)) != ""


def test_nothing_is_announced_from_a_checkout(installed, monkeypatch, tmp_path):
    # Re-installing over a checkout would replace a working copy with a tag.
    cache({"checked": time.time(), "latest": "0.3.0"})
    monkeypatch.setattr(update, "editable", lambda: tmp_path)
    assert update.notice() == ""


def test_the_check_never_runs_in_the_foreground(installed, monkeypatch):
    # The greeting reads the previous run's cache. Nothing here may block on a
    # request, however slow GitHub is being.
    started = []
    monkeypatch.setattr(update.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: started.append(kw)
    ))
    update.start()
    assert started and started[0]["daemon"] is True


def test_a_fresh_cache_starts_no_check(installed, monkeypatch):
    cache({"checked": time.time(), "latest": "0.3.0"})
    monkeypatch.setattr(update.threading, "Thread", lambda **kw: pytest.fail("checked anyway"))
    update.start()


def test_a_disabled_check_starts_no_thread(installed, monkeypatch):
    monkeypatch.setattr(update.threading, "Thread", lambda **kw: pytest.fail("checked anyway"))
    update.start(SimpleNamespace(update_check=False))


# -------------------------------------------------------------- coder update


def args(check: bool = False):
    return SimpleNamespace(check=check)


def test_updating_a_checkout_refuses_and_names_git_pull(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(update, "editable", lambda: tmp_path)
    monkeypatch.setattr(update, "_install", lambda tag: pytest.fail("installed over a checkout"))

    assert update.run(args()) == 1
    assert "git pull" in capsys.readouterr().out


def test_updating_installs_the_newest_tag(installed, monkeypatch, capsys):
    cache({"checked": time.time(), "latest": "0.3.0"})
    ran = []
    monkeypatch.setattr(update.subprocess, "call", lambda argv: ran.append(argv) or 0)

    assert update.run(args()) == 0
    assert ran == [["uv", "tool", "install", "--force", f"{update.GIT_URL}@v0.3.0"]]
    assert "0.3.0" in capsys.readouterr().out


def test_check_reports_without_installing(installed, monkeypatch, capsys):
    cache({"checked": time.time(), "latest": "0.3.0"})
    monkeypatch.setattr(update.subprocess, "call", lambda argv: pytest.fail("installed anyway"))

    assert update.run(args(check=True)) == 0
    out = capsys.readouterr().out
    assert "0.3.0" in out and "coder update" in out


def test_updating_when_current_says_so_and_stops(installed, monkeypatch):
    cache({"checked": time.time(), "latest": "0.2.0"})
    monkeypatch.setattr(update.subprocess, "call", lambda argv: pytest.fail("installed anyway"))
    assert update.run(args()) == 0


def test_updating_with_no_answer_explains_itself(installed, capsys):
    # `refresh` is the autouse no-op here, so the cache stays empty -- which is
    # what an unreachable API looks like from this side.
    assert update.run(args()) == 1
    assert "uv tool install" in capsys.readouterr().out


def test_updating_without_uv_says_which_command_is_missing(monkeypatch, capsys):
    def missing(argv):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(update.subprocess, "call", missing)
    assert update._install("0.3.0") == 1
    assert "uv" in capsys.readouterr().out
