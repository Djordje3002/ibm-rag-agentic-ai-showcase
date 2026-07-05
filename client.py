"""Course-compatible entry point for the Connoisseur MCP client."""

from ibm_rag_agentic_showcase.mcp_client import (
    call_tool,
    demo_get_restaurant_info,
    demo_get_review,
    demo_recommend_by_vibe,
    handle_sampling,
    list_roots,
    main,
    verify_connection,
)

__all__ = [
    "call_tool",
    "demo_get_restaurant_info",
    "demo_get_review",
    "demo_recommend_by_vibe",
    "handle_sampling",
    "list_roots",
    "verify_connection",
]

if __name__ == "__main__":
    main()
