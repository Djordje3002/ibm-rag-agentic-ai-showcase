"""Course-compatible entry point for the Connoisseur MCP host application."""

from ibm_rag_agentic_showcase.mcp_host import (
    SYSTEM_PROMPT,
    build_gradio_app,
    chat_with_agent,
    handle_chat,
    launch_app,
    make_model,
    run_react_agent,
)

__all__ = [
    "SYSTEM_PROMPT",
    "build_gradio_app",
    "chat_with_agent",
    "handle_chat",
    "launch_app",
    "make_model",
    "run_react_agent",
]

demo = build_gradio_app()

if __name__ == "__main__":
    launch_app(demo)
