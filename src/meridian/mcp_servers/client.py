"""Connect the MCP servers as LangChain tools.

The CLI uses stdio. Tests call the same FastMCP servers in-process.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from meridian.mcp_servers.data import build as build_data
from meridian.mcp_servers.history import build as build_history
from meridian.mcp_servers.lineage import build as build_lineage

_SERVERS = {
    "lineage": "meridian.mcp_servers.lineage",
    "data": "meridian.mcp_servers.data",
    "history": "meridian.mcp_servers.history",
}


class _DatasetQuery(BaseModel):
    dataset: str = Field(description="Dataset name.")


class _PartitionPath(BaseModel):
    path: str = Field(description="Parquet path.")


class _CompareQuery(BaseModel):
    path: str = Field(description="Parquet path.")
    profile_path: str = Field(description="Reference profile JSON path.")


class _SampleQuery(BaseModel):
    incident_id: str = Field(description="Incident id.")
    n: int = Field(description="Number of rows, from 1 through 20.")


class _EventQuery(BaseModel):
    dataset: str | None = Field(default=None, description="Dataset filter.")
    since: str | None = Field(default=None, description="Inclusive UTC lower bound.")
    until: str | None = Field(default=None, description="Inclusive UTC upper bound.")
    kind: str | None = Field(default=None, description="Event kind filter.")


class _EventId(BaseModel):
    event_id: str = Field(description="Change event id.")


def stdio_connections(root: Path) -> dict[str, StdioConnection]:
    """Stdio connection map for ``MultiServerMCPClient``."""
    env = dict(os.environ)
    env["MERIDIAN_EVIDENCE_ROOT"] = str(root)
    connections: dict[str, StdioConnection] = {}
    for name, module in _SERVERS.items():
        connections[name] = {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module],
            "env": env,
        }
    return connections


async def load_stdio_tools(root: Path) -> list[BaseTool]:
    """Load LangChain tools by spawning the three servers on stdio."""
    client = MultiServerMCPClient(cast("dict[str, Any]", stdio_connections(root)))
    return await client.get_tools()


def invoke_tool(tool: BaseTool, args: dict[str, Any]) -> str:
    """Call a tool and return its JSON text. Stdio tools are async-only."""
    try:
        raw = tool.invoke(args)
    except NotImplementedError:
        raw = asyncio.run(tool.ainvoke(args))
    return _tool_text(raw)


def inprocess_tools(root: Path) -> list[BaseTool]:
    """LangChain tools that call the FastMCP servers in this process."""
    os.environ["MERIDIAN_EVIDENCE_ROOT"] = str(root)
    servers = {
        "lineage": build_lineage(),
        "data": build_data(),
        "history": build_history(),
    }
    specs: list[tuple[FastMCP, str, str, type[BaseModel]]] = [
        (servers["lineage"], "get_upstream", "Datasets that feed this one.", _DatasetQuery),
        (servers["lineage"], "get_downstream", "Datasets fed by this one.", _DatasetQuery),
        (servers["lineage"], "get_contract", "Pandera contract summary.", _DatasetQuery),
        (servers["data"], "describe_partition", "Partition dtypes and null rates.", _PartitionPath),
        (
            servers["data"],
            "compare_to_reference",
            "PSI and null rate against a reference profile.",
            _CompareQuery,
        ),
        (servers["data"], "sample_failures", "Sample rows for an incident.", _SampleQuery),
        (servers["history"], "list_events", "Filter the change log.", _EventQuery),
        (servers["history"], "get_event", "Fetch one change event.", _EventId),
    ]
    tools: list[BaseTool] = []
    for server, name, description, schema in specs:
        tools.append(_bound(server, name, description, schema))
    return tools


def _bound(server: FastMCP, name: str, description: str, schema: type[BaseModel]) -> BaseTool:
    def _run(**kwargs: Any) -> str:
        result = asyncio.run(server.call_tool(name, kwargs))
        return _tool_text(result)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=schema,
    )


def _tool_text(result: object) -> str:
    """Normalize a tool result to the JSON text the agent records."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        text = _text_from_blocks(result)
        if text is not None:
            return text
    if isinstance(result, tuple):
        content = result[0]
        second = result[1] if len(result) > 1 else None
        text = _text_from_blocks(content)
        if text is not None:
            return text
        if isinstance(second, dict):
            return json.dumps(second, default=str)
        structured = getattr(second, "structured_content", None)
        if isinstance(structured, dict):
            return json.dumps(structured, default=str)
    if isinstance(result, dict):
        return json.dumps(result, default=str)
    raise TypeError("tool response was not a typed payload")


def _text_from_blocks(content: object) -> str | None:
    if isinstance(content, str) and content.startswith("{"):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str) and text.startswith("{"):
                return text
        text = getattr(first, "text", "")
        if isinstance(text, str) and text.startswith("{"):
            return text
    return None
