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
or call provider services. To check real credentials, endpoint availability,
and audio quality, run the console agent and listen to it.

Three branches exist, with different histories and purposes:

- `stepfun-agent`: the plugin plus one runnable agent. Nothing else. This is
  what someone forking the repository to build an agent starts from.
- `main`: the plugin, the agent, and the measurement tooling — evaluation
  harness, dataset replay, smoke test, timing probe.
- `add-stepfun-plugin`: based on LiveKit's upstream Git history, in its
  monorepo layout. This is the branch that becomes a PR.

The two layouts map as follows:

| Mainline | LiveKit PR branch |
| --- | --- |
| `livekit/plugins/stepfun/` | `livekit-plugins/livekit-plugins-stepfun/livekit/plugins/stepfun/` |
| Plugin metadata in `pyproject.toml` | `livekit-plugins/livekit-plugins-stepfun/pyproject.toml` |
| Plugin tests in `tests/test_stepfun_*.py` | `tests/test_plugin_stepfun_*.py` |
| `examples/`, `.env.example` | Not part of the PR |

Port plugin fixes to all three branches. Their repository histories differ;
copying the relevant paths keeps application files out of the upstream
contribution.
Retain the PR branch's short README, upstream project URLs, and workspace settings.
Update the upstream workspace lockfile when changing its dependencies.

In the LiveKit checkout, follow its `AGENTS.md` and
[contribution guide](https://github.com/livekit/agents/blob/main/CONTRIBUTING.md).
Run the StepFun tests in the LiveKit checkout with
`uv run pytest --plugin stepfun`. They require the plugin
dependencies, but no StepFun API key or network access.

Create a PR against `livekit/agents:main` from the PR branch after pushing it to
your fork. The local branch does not publish a package or create a GitHub PR.
