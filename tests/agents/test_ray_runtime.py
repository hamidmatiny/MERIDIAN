"""Workers reuse the driver environment instead of rebuilding it with uv."""

import os

from meridian.agents.ray_runtime import prepare_ray_runtime


def test_prepare_ray_runtime_disables_the_uv_rebuild() -> None:
    runtime = prepare_ray_runtime()
    import ray._private.ray_constants as constants

    assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False
    assert runtime == {"env_vars": {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}}
    assert "working_dir" not in runtime
    assert "py_executable" not in runtime
