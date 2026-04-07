"""Shared test helpers for langchain-collapse."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


def make_ai(tool_name: str, call_id: str) -> AIMessage:
    """Create an ``AIMessage`` with a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": call_id}],
    )


def make_tool(call_id: str, content: str = "result") -> ToolMessage:
    """Create a ``ToolMessage`` for a given call ID."""
    return ToolMessage(content=content, tool_call_id=call_id)


def make_read_pairs(n: int) -> list[AnyMessage]:
    """Create *n* consecutive ``read_file`` (AIMessage, ToolMessage) pairs."""
    msgs: list[AnyMessage] = []
    for i in range(n):
        cid = f"tc{i}"
        msgs.append(make_ai("read_file", cid))
        msgs.append(make_tool(cid, f"content of file {i}"))
    return msgs
