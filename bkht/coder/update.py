"""Noticing that a newer release exists, and installing it when asked.

Releases are git tags. There is no package index in the middle: `install.sh`
and this module both resolve the newest ``vX.Y.Z`` and hand it to the same
`uv tool install --force <repo>@<tag>` the installer has always run, so an
update is exactly a re-install of a named ref.

Two rules shape everything below.

**The check never blocks a session.** The greeting reads a cache written by an
earlier run; the refresh that fills it runs on a daemon thread and is allowed
to lose. There is deliberately no path where a slow or unreachable GitHub
delays a prompt, and no failure here is ever fatal -- an update check is the
least important thing this program does.

**The check is the one outbound request coder makes that is not to your own
Ollama.** It asks GitHub for a version number once a day and sends nothing;
no code, no prompt, no telemetry. It is still a request, so it is written down
in the README, reported by `coder doctor`, off for anything that is not an
interactive terminal, and switched off entirely by `update_check false`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx

from . import terminal
from .doctor import _is_this_project, running_from, version
from .session import STATE_DIR

REPO = "https://github.com/thebkht/bkht-coder"
GIT_URL = f"git+{REPO}.git"
LATEST_URL = "https://api.github.com/repos/thebkht/bkht-coder/releases/latest"

CACHE = STATE_DIR / "update.json"

#: How long a cached answer stands. A day is short enough that a release is
#: noticed within one, and long enough that the request is invisible.
INTERVAL = 24 * 60 * 60

#: Short on purpose. This runs off the critical path, but a thread hanging on a
#: dead socket for a minute still outlives the session that started it.
TIMEOUT = 3.0

#: A release, and the dev versions leading to it: 1.2, v1.2.3, 1.2.3.dev4+gabc.
_VERSION = re.compile(r"^[vV]?(\d+(?:\.\d+)*)(.*)$")


def parse(text: str) -> tuple | None:
    """A key that orders two versions, or ``None`` when it cannot.

    Small on purpose. The only runtime dependency this project has is httpx,
    and adding `packaging` to compare two dotted numbers would be the largest
    thing in the wheel that the program itself never uses.

    What has to come out right is that a dev version sorts *below* the release
    it leads to -- ``0.3.0.dev4+g9ab24a6`` is not 0.3.0, it is on the way to it
    -- because a checkout otherwise reports itself as current the day before
    the tag it anticipates is actually cut.
    """
    match = _VERSION.match((text or "").strip())
    if match is None:
        return None
    numbers = tuple(int(part) for part in match.group(1).split("."))
    # Pad so 0.3 and 0.3.0 compare equal rather than by length.
    numbers = (numbers + (0, 0, 0))[:3]
    # 1 for a bare release, 0 for anything trailing it. Only ever a tiebreak
    # between the same three numbers, which is the dev-versus-release case.
    return numbers, 1 if not match.group(2) else 0


def newer(candidate: str, than: str) -> bool:
    """Whether ``candidate`` is a release worth telling somebody about.

    Unparseable on either side means the honest answer is "cannot tell", and
    the honest thing to do with that is stay quiet.
    """
    left, right = parse(candidate), parse(than)
    if left is None or right is None:
        return False
    return left > right


def _read() -> dict:
    """The cache, or an empty one. Never raises: this is a convenience file."""
    try:
        loaded = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write(stored: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(stored), encoding="utf-8")
    except OSError:
        pass


def stale(now: float | None = None) -> bool:
    """Whether the cached answer is old enough to ask again."""
    now = time.time() if now is None else now
    checked = _read().get("checked")
    if not isinstance(checked, (int, float)):
        return True
    return now - checked >= INTERVAL


def cached() -> str | None:
    """The newest release an earlier run heard about, without asking again."""
    latest = _read().get("latest")
    return latest if isinstance(latest, str) and latest else None


def refresh() -> None:
    """Ask GitHub for the newest release and cache the answer.

    Every failure is swallowed -- offline, rate-limited, a shape the API has
    never returned before -- but the attempt is *always* stamped. Without that
    stamp a machine with no network would retry on every single launch, which
    is the one way a check this unimportant could become expensive.
    """
    stored = dict(_read())
    stored["checked"] = time.time()
    try:
        response = httpx.get(
            LATEST_URL,
            timeout=TIMEOUT,
            headers={"Accept": "application/vnd.github+json"},
            follow_redirects=True,
        )
        response.raise_for_status()
        tag = response.json().get("tag_name")
        if isinstance(tag, str) and parse(tag) is not None:
            stored["latest"] = tag.lstrip("vV")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    _write(stored)


def editable() -> Path | None:
    """The checkout this is running out of, when it is running out of one.

    `uv tool install` copies the package into an environment of its own; a
    checkout does not. Re-installing over the second would replace somebody's
    working copy with a tag, which is not an update, it is a loss -- so this is
    what both the notice and `coder update` refuse on.
    """
    origin = running_from()
    return origin if _is_this_project(origin) else None


def enabled(settings=None, stream=None) -> bool:
    """Whether this install should be checking for releases at all.

    Three ways to be off, and all of them live here rather than at the call
    sites, so there is one answer to "why did it not say anything".
    """
    if settings is not None and not getattr(settings, "update_check", True):
        return False
    if not terminal.interactive(stream):
        return False
    return editable() is None


def available() -> str | None:
    """The cached release, when it is newer than what is running."""
    latest = cached()
    if latest is None:
        return None
    return latest if newer(latest, version()) else None


def start(settings=None) -> None:
    """Refresh the cache in the background, if it is time and we are allowed.

    A daemon thread because the answer is for the *next* run: nothing waits on
    this one, and a session ending mid-request should end.
    """
    if not enabled(settings) or not stale():
        return
    threading.Thread(target=refresh, name="update-check", daemon=True).start()


def notice(settings=None, stream=None) -> str:
    """The one line the greeting adds, or nothing at all."""
    if not enabled(settings, stream):
        return ""
    latest = available()
    return f"v{latest} available · coder update" if latest else ""


#: uv verifies TLS against roots it bundles itself, so a proxy or antivirus
#: that re-signs HTTPS -- the ordinary managed-laptop setup -- fails with
#: `invalid peer certificate: UnknownIssuer` on a machine where git, curl and
#: the browser all work. This tells uv to trust the platform's certificate
#: store instead.
#:
#: Set as an environment variable rather than passed as a flag, because a flag
#: uv does not know is a hard error that would replace the real failure with a
#: worse one. Which variable is asked of uv itself: the setting was renamed
#: from `--native-tls` to `--system-certs`, and a uv new enough to have the
#: second warns about the first -- so setting both would put a scolding in the
#: middle of an install that is already going badly.
NEW_CERTS, OLD_CERTS = "UV_SYSTEM_CERTS", "UV_NATIVE_TLS"


def _certs_variable() -> str:
    """Whichever spelling this uv understands, its own help being the authority."""
    try:
        help_text = subprocess.run(
            ["uv", "tool", "install", "--help"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        ).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return NEW_CERTS
    return NEW_CERTS if "--system-certs" in help_text else OLD_CERTS

TLS_HELP = """\
That is uv failing to verify TLS, not coder failing to build. It usually means
a proxy or antivirus is re-signing HTTPS on this machine, and uv checks against
its own bundled roots rather than the system store.

Retried with the system store already. If it still fails:
  * install from a Python you already have, so uv downloads no interpreter:
      uv tool install --force --no-python-downloads {target}
  * or point uv at your certificate bundle:
      SSL_CERT_FILE=<path to your CA bundle>"""


def _install(tag: str) -> int:
    """Re-install from a tag, showing the work.

    `uv tool install --force` is what the installer runs, so an update lands a
    user in exactly the state a fresh install would have.

    A failure is retried once against the platform's certificate store. The
    retry is unconditional because the output is streamed rather than captured
    -- reading the reason would mean hiding the progress of an install that
    takes minutes -- and a second attempt on a network that was never the
    problem costs one repeated error message.
    """
    target = f"{GIT_URL}@v{tag}"
    argv = ["uv", "tool", "install", "--force", target]
    print(f"Installing {target}")
    try:
        code = subprocess.call(argv)
        if code == 0:
            return 0

        print("\nRetrying with this machine's certificate store.")
        code = subprocess.call(argv, env={**os.environ, _certs_variable(): "1"})
        if code != 0:
            print("\n" + TLS_HELP.format(target=target))
        return code
    except FileNotFoundError:
        print("uv is not on PATH. See https://docs.astral.sh/uv/ to install it.")
        return 1


def run(args) -> int:
    """`coder update`: check, and unless --check, install what is there.

    Unlike the background check this one is synchronous and says what happened
    at every branch -- it was typed, so silence would read as a hang.
    """
    checkout = editable()
    if checkout is not None:
        print(
            f"coder is running from a checkout at {checkout}, not an installed copy.\n"
            "Update it with `git pull` -- re-installing over it would replace your\n"
            "working copy with a release."
        )
        return 1

    refresh()
    latest = cached()
    if latest is None:
        print(
            "Could not reach the releases API. Check your connection, or install\n"
            f"a release directly:  uv tool install --force {GIT_URL}@vX.Y.Z"
        )
        return 1

    running = version()
    if not newer(latest, running):
        print(f"coder {running or 'unknown version'} is the newest release.")
        return 0

    print(f"coder {running or 'unknown version'} is installed; v{latest} is available.")
    if args.check:
        print("Run `coder update` to install it.")
        return 0
    return _install(latest)
