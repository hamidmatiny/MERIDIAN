# ADR-015: MCP for agent tools, read-only, typed, and ref-stamped

- Status: Accepted

## Context

Root cause needs lineage, the partition, and the change log. Those reads should be tools with a schema, not functions the model calls by importing the repo. The tool surface has to stay read-only so a hypothesis cannot edit the evidence it is judged against.

## Decision

Three MCP servers live in `src/meridian/mcp_servers/`, built with FastMCP from the official `mcp` package. `langchain-mcp-adapters` 0.3.1 connects them. That adapter imports `RequestContext` from `mcp.shared.context`, which MCP 2 removed when it renamed FastMCP to `MCPServer`. This phase pins `mcp>=1.12,<2` (resolved to 1.30) so FastMCP and the adapter import together.

Tools:

| Server | Tool | Ref |
| --- | --- | --- |
| lineage | `get_upstream` | `lineage:upstream:<dataset>` |
| lineage | `get_downstream` | `lineage:downstream:<dataset>` |
| lineage | `get_contract` | `contract:<dataset>` |
| data | `describe_partition` | `partition:<path>` and `column:<path>:<column>` |
| data | `compare_to_reference` | `psi:<dataset>` and `psi:<dataset>:<column>` |
| data | `sample_failures` | `sample:<incident_id>`, at most 20 rows |
| history | `list_events` | `events:<dataset>` and `event:<id>` on each event |
| history | `get_event` | `event:<id>` |

Each tool returns a Pydantic model. FastMCP serializes it to JSON. A non-finite PSI is stored as null. The CLI loads the servers with stdio and `MultiServerMCPClient`. Tests call the same FastMCP servers in-process, because the adapter has no in-process transport. Stdio tools are async-only, so `invoke_tool` falls back to `ainvoke` and keeps the JSON text.

`MERIDIAN_EVIDENCE_ROOT` points the data and history servers at `events.jsonl` and `incidents/<id>.json`. The root-cause node writes the incident file before the loop so `sample_failures` can read it.

## Consequences

- A tool cannot write the log, the partition, or the incident. Tests hash the evidence directory before and after the calls.
- The MCP major pin is deliberate. Moving to MCP 2 requires an adapter that imports the renamed server types.
- Every citation the model is allowed to make has to be one of these refs. ADR-016 enforces that.

## Alternatives considered

- Plain Python functions registered only as LangChain tools. Rejected. The phase asks for MCP, and the same servers have to be reachable from a later process.
- MCP 2 with `MCPServer`. Rejected for this phase because `langchain-mcp-adapters` 0.3.1 does not import against it.
- An in-memory MultiServerMCPClient transport. Rejected because that client does not offer one. In-process tests call `FastMCP.call_tool` instead.
