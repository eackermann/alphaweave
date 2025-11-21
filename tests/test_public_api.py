"""Tests for public API exposure."""

def test_public_api():
    import alphaweave as aw

    # Frame should be exposed at top level
    assert hasattr(aw, "Frame")

    # Namespaces should exist
    for ns in ["core", "data", "engine", "strategy", "results", "utils"]:
        assert hasattr(aw, ns), f"{ns} not in public API"

