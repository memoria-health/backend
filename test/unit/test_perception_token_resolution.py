"""The identity service must resolve perception's bearer the same way
perception itself does: environment first, then the file on the shared
volume.

The file path is the whole point of the design — a node's env is fixed
at container start, so writing `/persist/secrets/perception_token` is
how auth gets switched on without recycling the pod. When only the
SERVER honoured the file, doing exactly that locked out the one client
that is supposed to get in: every `/observe_raw` returned 401, every
tracking step raised, and the service kept answering `/state` as
`ready` with an empty camera map. Nothing upstream reads 401s, so it
surfaced only as "identity never locates anybody" (2026-08-23).
"""

import pytest

from identity.pod_client import load_perception_token


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("NEMARA_PERCEPTION_TOKEN", raising=False)
    monkeypatch.delenv("NEMARA_PERCEPTION_TOKEN_FILE", raising=False)


def test_the_environment_wins_when_set(monkeypatch, tmp_path):
    f = tmp_path / "tok"
    f.write_text("from-file")
    monkeypatch.setenv("NEMARA_PERCEPTION_TOKEN", "from-env")
    monkeypatch.setenv("NEMARA_PERCEPTION_TOKEN_FILE", str(f))
    assert load_perception_token() == "from-env"


def test_the_file_is_read_when_the_environment_is_unset(monkeypatch, tmp_path):
    """The regression. This is the live pod's configuration."""
    f = tmp_path / "tok"
    f.write_text("from-file\n")          # trailing newline is normal
    monkeypatch.setenv("NEMARA_PERCEPTION_TOKEN_FILE", str(f))
    assert load_perception_token() == "from-file"


def test_no_env_and_no_file_means_no_auth(monkeypatch, tmp_path):
    """Empty disables auth on the server too, so dev stays open."""
    monkeypatch.setenv("NEMARA_PERCEPTION_TOKEN_FILE",
                       str(tmp_path / "absent"))
    assert load_perception_token() == ""


def test_the_client_sends_the_file_token_as_a_bearer(monkeypatch, tmp_path):
    """Resolution is only half of it — it has to reach the wire."""
    from identity.pod_client import SamServiceClient

    f = tmp_path / "tok"
    f.write_text("shared-secret")
    monkeypatch.setenv("NEMARA_PERCEPTION_TOKEN_FILE", str(f))
    client = SamServiceClient(base_url="http://127.0.0.1:8000", timeout_s=5.0)
    assert (client._client.headers["Authorization"]
            == "Bearer shared-secret")
