"""Connect the MCP servers as LangChain tools.

The CLI opens one stdio session per server for a graph run and shares it.
Tests call the same FastMCP servers in-process.
"""

import asyncio
import concurrent.futures
import contextlib
import json
import os
import sys
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from mcp import ClientSession
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
    n: int = Field(description="Number of rows, from 1 through 10.")


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


_Job = tuple[Callable[[], Awaitable[object]], concurrent.futures.Future[object]]


class StdioSession:
    """One stdio process per server, shared across every tool call in the run.

    Sessions stay on a single event-loop thread. Tool calls are awaited on that
    same task, then the processes are closed when the context exits.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="meridian-mcp", daemon=True)
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._error: BaseException | None = None
        self._queue: asyncio.Queue[_Job | None] | None = None
        self._open_for_calls = False
        self.tools: list[BaseTool] = []

    def __enter__(self) -> "StdioSession":
        self._thread.start()
        if not self._opened.wait(timeout=30):
            self.close()
            raise TimeoutError("MCP stdio session did not open")
        if self._error is not None:
            error = self._error
            self.close()
            raise error
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the session thread and close the server processes."""
        self._open_for_calls = False
        if self._thread.ident is None:
            return
        queue = self._queue
        if queue is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(queue.put_nowait, None)
        self._closed.wait(timeout=15)

    def call(self, factory: Callable[[], Awaitable[object]]) -> object:
        """Run ``factory`` on the session task and return its result."""
        if not self._open_for_calls or self._queue is None or self._loop.is_closed():
            raise RuntimeError("MCP session is closed")
        box: concurrent.futures.Future[object] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (factory, box))
        return box.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:
            if self._error is None:
                self._error = exc
            self._opened.set()
        finally:
            if not self._loop.is_closed():
                self._loop.close()
            self._closed.set()

    async def _serve(self) -> None:
        queue: asyncio.Queue[_Job | None] = asyncio.Queue()
        self._queue = queue
        client = MultiServerMCPClient(cast("dict[str, Any]", stdio_connections(self._root)))
        try:
            async with contextlib.AsyncExitStack() as stack:
                loaded = await _load_sessions(client, stack, self)
                self.tools = loaded
                self._open_for_calls = True
                self._opened.set()
                await _drain(queue)
        except BaseException as exc:
            self._error = exc
            self._opened.set()
        finally:
            self._open_for_calls = False


async def _load_sessions(
    client: MultiServerMCPClient,
    stack: contextlib.AsyncExitStack,
    owner: StdioSession,
) -> list[BaseTool]:
    loaded: list[BaseTool] = []
    for name in client.connections:
        session = await stack.enter_async_context(client.session(name))
        for tool in await load_mcp_tools(session, server_name=name):
            loaded.append(_sync_tool(tool, owner, session))
    return loaded


async def _drain(queue: asyncio.Queue[_Job | None]) -> None:
    while True:
        job = await queue.get()
        if job is None:
            return
        factory, box = job
        if box.done():
            continue
        try:
            box.set_result(await factory())
        except BaseException as exc:
            box.set_exception(exc)


def _sync_tool(tool: BaseTool, owner: StdioSession, session: ClientSession) -> BaseTool:
    """Expose one MCP tool as a sync call on the shared session."""
    name = tool.name

    def _run(**kwargs: Any) -> str:
        payload = dict(kwargs)

        def _call() -> Awaitable[object]:
            return session.call_tool(name, payload)

        return _tool_text(owner.call(_call))

    return StructuredTool(
        name=name,
        description=tool.description or name,
        args_schema=cast("Any", tool.args_schema),
        func=_run,
    )


def _observe_tool(name: str, text: str) -> None:
    from meridian.audit.store import note
    from meridian.tracing import note_tool

    refs = text.count("event:")
    note_tool(name, refs)
    note("tool_calls", {"tool": name, "ref_count": refs})


def invoke_tool(tool: BaseTool, args: dict[str, Any]) -> str:
    """Call a tool and return its JSON text."""
    try:
        raw = tool.invoke(args)
    except NotImplementedError:
        raw = asyncio.run(tool.ainvoke(args))
    text = _tool_text(raw)
    _observe_tool(str(getattr(tool, "name", "tool")), text)
    return text


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
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return json.dumps(structured, default=str)
    content = getattr(result, "content", None)
    text = _text_from_blocks(content)
    if text is not None:
        return text
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
