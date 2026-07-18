"""Integration-test starter.

TODO: use httpx.ASGITransport and mock VLLMClient.complete/stream so CI does not require a GPU.
Then add a separate @pytest.mark.gpu test that talks to a live vLLM server.
"""


def test_integration_placeholder() -> None:
    assert True
