import pytest


@pytest.fixture(autouse=True)
def stepfun_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's provider credentials."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)
