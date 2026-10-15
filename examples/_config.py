"""Configuration loading for the examples."""

from __future__ import annotations

import os
import pathlib

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit(
        "python-dotenv is required by the examples.\n"
        "    uv pip install python-dotenv\n"
        "or\n"
        "    pip install 'livekit-plugins-stepfun[examples]'"
    ) from exc

_ENV_FILES = (".env.local", ".env")


def load_env(*, verbose: bool = False) -> list[pathlib.Path]:
    """Load whichever env files exist, from the cwd and the project root.

    Checking both means the examples work whether they are run from the
    repository root or from inside `examples/`.
    """
    found: list[pathlib.Path] = []
    roots = [pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parent.parent]
    seen: set[pathlib.Path] = set()

    for root in roots:
        for name in _ENV_FILES:
            path = (root / name).resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            # override=False so an exported variable is never silently
            # discarded, and so earlier files win over later ones.
            load_dotenv(path, override=False)
            found.append(path)

    if verbose:
        if found:
            for path in found:
                print(f"config: loaded {path}")
        else:
            print("config: no .env.local or .env found; using the environment only")
    return found


def get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def get_bool(name: str, default: bool) -> bool:
    value = get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def require(name: str, *, hint: str = "") -> str:
    value = get(name)
    if value:
        return value
    extra = f" {hint}" if hint else ""
    raise SystemExit(
        f"{name} is not set.\n  Copy .env.example to .env and fill it in, or export {name}.{extra}"
    )
