import pytest


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path_factory, monkeypatch):
    """Keep plans/events out of the real ~/.local/state during tests."""
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path_factory.mktemp("state")))
