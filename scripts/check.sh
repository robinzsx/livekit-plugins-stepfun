#!/usr/bin/env bash
# Run ruff, mypy and pytest against every livekit-agents version present.
#
# The plugin supports both the 1.x and 2.x lines with no version branching, and
# the only way that stays true is checking both on every change. Each venv is
# invoked by path, so nothing has to be activated and the two cannot be confused.
#
#   ./scripts/check.sh            # every .venv* found
#   ./scripts/check.sh .venv1x    # just one
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

venvs=("${@:-}")
if [ -z "${venvs[0]:-}" ]; then
    venvs=()
    for candidate in .venv*; do
        [ -x "$candidate/bin/python" ] && venvs+=("$candidate")
    done
fi

if [ ${#venvs[@]} -eq 0 ]; then
    echo "no virtualenvs found; create one with: uv venv .venv && uv pip install -e '.[dev,llm]'"
    exit 1
fi

failed=0
for venv in "${venvs[@]}"; do
    if [ ! -x "$venv/bin/python" ]; then
        echo "skipping $venv (no interpreter)"
        continue
    fi

    version=$("$venv/bin/python" -c 'import livekit.agents as a; print(a.__version__)' 2>/dev/null) \
        || version="livekit-agents missing"
    printf '\n=== %s (livekit-agents %s) ===\n' "$venv" "$version"

    for step in \
        "ruff:$venv/bin/ruff check livekit tests examples" \
        "mypy:$venv/bin/mypy -p livekit.plugins.stepfun" \
        "pytest:$venv/bin/pytest -q"
    do
        name=${step%%:*}
        cmd=${step#*:}
        printf '%-7s ' "$name"
        if output=$(eval "$cmd" 2>&1); then
            echo "ok    $(echo "$output" | tail -1)"
        else
            echo "FAIL"
            echo "$output" | tail -25
            failed=1
        fi
    done
done

printf '\n'
if [ $failed -eq 0 ]; then
    echo "all checks passed"
else
    echo "some checks failed"
fi
exit $failed
