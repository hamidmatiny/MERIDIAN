"""CLI fix model. It chooses a kind from the incident, not from a label file."""

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from meridian.agents.fix import PatchDraft
from meridian.contracts.fault_type import FaultType
from meridian.contracts.remediation import RemediationKind
from meridian.graph.limits import escalate_source

_USAGE = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}

_REPAIR = """\
def repair(df):
    out = df.copy()
    for name in list(out.columns):
        if "speed" in str(name):
            high = out[name] > 60
            out.loc[high, name] = out.loc[high, name] / 3.6
        if name == "total_distance_km":
            high = out[name] > 2000
            out.loc[high, name] = out.loc[high, name] / 1000.0
        if str(name).endswith("_renamed"):
            out = out.rename(columns={name: str(name)[: -len("_renamed")]})
    return out.drop_duplicates()
"""

_QUARANTINE = """\
def quarantine(df):
    return df.isna().any(axis=1)
"""

_CHECK_REPAIR = """\
def check(df):
    import pandera as pa
    columns = {}
    for name in df.columns:
        if "speed" in str(name):
            columns[name] = pa.Column(float, pa.Check.le(60))
        if name == "total_distance_km":
            columns[name] = pa.Column(float, pa.Check.le(2000))
    if columns:
        pa.DataFrameSchema(columns).validate(df)
"""

_CHECK_NULLS = """\
def check(df):
    import pandera as pa
    if int(df.isna().sum().sum()) != 0:
        raise ValueError("nulls remain")
    pa.DataFrameSchema({}).validate(df)
"""

_CHECK_PRESENT = """\
def check(df):
    import pandera as pa
    pa.DataFrameSchema({}).validate(df)
"""


class _Bound:
    def __init__(self, parent: "HeuristicFixModel", *, structured: bool) -> None:
        self._parent = parent
        self._structured = structured

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> object:
        del kwargs
        if self._structured:
            return self._parent._structured(messages)
        return AIMessage(content="done", usage_metadata=_USAGE)


class HeuristicFixModel:
    """Pick a remediation kind from the incident's fault type.

    The choice does not open a label file. Null explosions are quarantined.
    A late partition is a rerun. A unit, schema, or key fault is a repair.
    """

    def __init__(self) -> None:
        self.model_name = "heuristic-fix"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> _Bound:
        del tools, kwargs
        return _Bound(self, structured=False)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> _Bound:
        del schema, include_raw, kwargs
        return _Bound(self, structured=True)

    def _structured(self, messages: list[BaseMessage]) -> dict[str, object]:
        draft = _draft(_fault(messages))
        raw = AIMessage(content=draft.model_dump_json(), usage_metadata=_USAGE)
        return {"raw": raw, "parsed": draft, "parsing_error": None}


def _draft(fault: str | None) -> PatchDraft:
    if fault == FaultType.NULL_EXPLOSION.value:
        return PatchDraft(
            remediation_kind=RemediationKind.QUARANTINE.value,
            module_source=_QUARANTINE,
            regression_test=_CHECK_NULLS,
        )
    if fault == FaultType.LATE_PARTITION.value:
        return PatchDraft(
            remediation_kind=RemediationKind.RERUN_UPSTREAM.value,
            module_source=escalate_source(),
            regression_test=_CHECK_PRESENT,
        )
    if fault in {
        FaultType.UNIT_MISMATCH.value,
        FaultType.SCHEMA_DRIFT.value,
        FaultType.DUPLICATE_KEYS.value,
    }:
        return PatchDraft(
            remediation_kind=RemediationKind.DATA_REPAIR.value,
            module_source=_REPAIR,
            regression_test=_CHECK_REPAIR,
        )
    return PatchDraft(
        remediation_kind=RemediationKind.ESCALATE.value,
        module_source=escalate_source(),
        regression_test=_CHECK_PRESENT,
    )


def _fault(messages: list[BaseMessage]) -> str | None:
    text = "\n".join(message.content for message in messages if isinstance(message.content, str))
    match = re.search(r'"fault_type"\s*:\s*"([^"]+)"', text)
    if match is None:
        return None
    return match.group(1)
