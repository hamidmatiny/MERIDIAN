"""Lineage tools: upstream, downstream, and the pandera contract summary."""

from mcp.server.fastmcp import FastMCP

from meridian.contracts.tool_results import ContractSummary, LineageResult
from meridian.mcp_servers.impl import contract_summary, lineage_result


def build() -> FastMCP:
    """Build the lineage server."""
    mcp = FastMCP("meridian-lineage")

    @mcp.tool()
    def get_upstream(dataset: str) -> LineageResult:
        """Datasets that feed this one, nearest first."""
        return lineage_result(dataset, direction="upstream")

    @mcp.tool()
    def get_downstream(dataset: str) -> LineageResult:
        """Datasets fed by this one, nearest first."""
        return lineage_result(dataset, direction="downstream")

    @mcp.tool()
    def get_contract(dataset: str) -> ContractSummary:
        """Pandera column names and named checks for this dataset."""
        return contract_summary(dataset)

    return mcp


def main() -> None:
    """Serve the lineage tools on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
