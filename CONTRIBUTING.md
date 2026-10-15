# Contributing

Use Python 3.10+ and install the development and example dependencies:

```bash
uv sync --extra dev --extra examples
./scripts/check.sh .venv
```

The checks cover Ruff lint and formatting, strict mypy for the plugin, and
offline pytest tests. Pass additional virtualenv paths to check other Python
or LiveKit versions. Invalid environment paths fail the check.

Keep protocol tests that verify request payloads, transcript/audio handling,
error translation, option snapshots, and resource cleanup. Shared scripted
WebSockets live in `tests/fake_ws.py`. Tests must not load a developer's `.env`
or call provider services. Use `examples/smoke_test.py` separately to check
real credentials, endpoint availability, and audio quality.

The `main` branch owns the standalone package, examples, and user documentation.
The `add-stepfun-plugin` branch is based on LiveKit's upstream Git history and
uses its monorepo layout:

| Mainline | LiveKit PR branch |
| --- | --- |
| `livekit/plugins/stepfun/` | `livekit-plugins/livekit-plugins-stepfun/livekit/plugins/stepfun/` |
| Plugin metadata in `pyproject.toml` | `livekit-plugins/livekit-plugins-stepfun/pyproject.toml` |
| Plugin tests in `tests/test_stepfun_*.py` | `tests/stepfun/test_stepfun_*.py` (excluding application session tests) |
| `examples/`, example tests, `.env.example` | Remain on mainline |

Port plugin fixes to both branches. Their repository histories differ; copying
the relevant paths keeps application files out of the upstream contribution.
Retain the PR branch's short README, upstream project URLs, and workspace settings.
Update the upstream workspace lockfile when changing its dependencies.

In the LiveKit checkout, follow its `AGENTS.md` and
[contribution guide](https://github.com/livekit/agents/blob/main/CONTRIBUTING.md).
Run the StepFun tests with `uv run pytest tests/stepfun --plugin stepfun`.
They require the plugin dependencies, but no StepFun API key or network access.

Create a PR against `livekit/agents:main` from the PR branch after pushing it to
your fork. The local branch does not publish a package or create a GitHub PR.
