"""History tools over the seeded events.jsonl log."""

from mcp.server.fastmcp import FastMCP

from meridian.contracts.tool_results import EventPage, EventView
from meridian.mcp_servers.impl import evidence_root_from_env, get_change_event, list_change_events


def build() -> FastMCP:
    """Build the history server."""
    mcp = FastMCP("meridian-history")

    @mcp.tool()
    def list_events(
        dataset: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
    ) -> EventPage:
        """Change events, optionally filtered by dataset, time, and kind."""
        return list_change_events(
            evidence_root_from_env(),
            dataset=dataset,
            since=since,
            until=until,
            kind=kind,
        )

    @mcp.tool()
    def get_event(event_id: str) -> EventView:
        """One change event by id."""
        return get_change_event(evidence_root_from_env(), event_id)

    return mcp


def main() -> None:
    """Serve the history tools on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
