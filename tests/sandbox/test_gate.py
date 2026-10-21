"""Every banned construct is rejected before a remediation can run."""

import pytest

from meridian.sandbox.gate import check_source

_BANNED = [
    "import os\n",
    "import sys\n",
    "import subprocess\n",
    "import socket\n",
    "import pathlib\n",
    "import importlib\n",
    "import os.path\n",
    "from os import path\n",
    "from pathlib import Path\n",
    "open('x')\n",
    "exec('x')\n",
    "eval('1')\n",
    "compile('1', '<s>', 'exec')\n",
    "__import__('os')\n",
    "x = __builtins__\n",
    "value.__dict__\n",
    "getattr(value, '__class__')\n",
    "getattr(value, name)\n",
    "from . import helper\n",
]


@pytest.mark.parametrize("source", _BANNED)
def test_banned_construct_is_rejected(source: str) -> None:
    assert check_source(source)


def test_each_context_reads_the_config_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    from meridian.config import REGRESSION_MODULES, REPAIR_MODULES

    assert REPAIR_MODULES == ("pandas", "numpy", "math", "datetime")
    assert REGRESSION_MODULES == ("pandas", "numpy", "math", "datetime", "pandera")
    monkeypatch.setattr("meridian.sandbox.gate.REPAIR_MODULES", ("math",))
    monkeypatch.setattr("meridian.sandbox.gate.REGRESSION_MODULES", ("math", "pandera"))
    assert check_source("import pandas\n")
    assert check_source("import math\n") == []
    assert check_source("import pandera\n")
    assert check_source("import pandera\n", allow_pandera=True) == []
    assert check_source("import math\n", allow_pandera=True) == []


def test_allowlisted_import_and_pandera_split() -> None:
    source = "import pandas as pd\nimport math\nfrom datetime import datetime\n"
    assert check_source(source) == []
    pandera = "import pandera as pa\n"
    assert check_source(pandera)
    assert check_source(pandera, allow_pandera=True) == []


def test_builtins_attribute_and_non_string_getattr_are_rejected() -> None:
    assert check_source("__builtins__.open\n")
    assert check_source("getattr(value, 1)\n")
    assert check_source("from pandas import DataFrame\n") == []


def test_syntax_error_is_a_rejection() -> None:
    reasons = check_source("def repair(:\n")
    assert reasons
    assert reasons[0].startswith("syntax error")
