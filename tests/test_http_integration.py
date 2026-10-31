"""Live HTTP tests require the optional Stash simulation environment."""

import pytest

pytestmark = pytest.mark.skip(reason="Run the maintained integration harness before testing a live Stash instance.")


def test_http_integration_placeholder():
    """Keep the integration test target explicit without contacting a real Stash."""
    assert True
