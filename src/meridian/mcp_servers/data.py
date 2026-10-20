"""Data tools: partition shape, PSI against a reference, and incident samples."""

from mcp.server.fastmcp import FastMCP

from meridian.contracts.tool_results import FailureSample, PartitionDescription, ReferenceComparison
from meridian.mcp_servers import impl


def build() -> FastMCP:
    """Build the data server."""
    mcp = FastMCP("meridian-data")

    @mcp.tool()
    def describe_partition(path: str) -> PartitionDescription:
        """Column dtypes and null rates for a parquet partition."""
        return impl.describe_partition(path)

    @mcp.tool()
    def compare_to_reference(path: str, profile_path: str) -> ReferenceComparison:
        """Per-column PSI and null rate against a reference profile."""
        return impl.compare_to_reference(path, profile_path)

    @mcp.tool()
    def sample_failures(incident_id: str, n: int) -> FailureSample:
        """Up to 20 sample rows stored for an incident."""
        return impl.sample_failures(impl.evidence_root_from_env(), incident_id, n)

    return mcp


def main() -> None:
    """Serve the data tools on stdio."""
    build().run(transport="stdio")


if __name__ == "__main__":
    main()
