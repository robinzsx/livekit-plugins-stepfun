#!/usr/bin/env bash
# Check the supplied virtualenvs, or every .venv* in this checkout.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

venvs=("$@")
if [ ${#venvs[@]} -eq 0 ]; then
    for candidate in .venv*; do
        [ -x "$candidate/bin/python" ] && venvs+=("$candidate")
    done
fi
if [ ${#venvs[@]} -eq 0 ]; then
    echo "No virtualenvs found. Run: uv sync --extra dev --extra examples"
    exit 1
fi

failed=0
for venv in "${venvs[@]}"; do
    if [ ! -x "$venv/bin/python" ]; then
        echo "Missing interpreter: $venv/bin/python"
        failed=1
        continue
    fi
    "$venv/bin/python" -c 'import sys, livekit.agents as a; print(sys.version.split()[0], "livekit-agents", a.__version__)' || failed=1
    "$venv/bin/python" -m ruff check livekit tests examples || failed=1
    "$venv/bin/python" -m ruff format --check livekit tests examples || failed=1
    "$venv/bin/python" -m mypy -p livekit.plugins.stepfun || failed=1
    "$venv/bin/python" -m pytest -q || failed=1
done
exit "$failed"
