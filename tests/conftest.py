"""Live tests get a longer timeout than the unit-test default."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(pytest.mark.timeout(1200))
