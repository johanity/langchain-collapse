"""Collapse consecutive read/search tool-call groups to save context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

__all__ = ["CollapseMiddleware"]

_DEFAULT_COLLAPSE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "grep",
        "glob",
        "web_search",
    }
)

_DEFAULT_MIN_GROUP_SIZE: int = 2


def _find_collapsible_groups(
    messages: list[AnyMessage],
    collapse_tools: frozenset[str],
    min_group_size: int,
) -> list[tuple[int, int]]:
    """Identify ranges of consecutive collapsible tool-call pairs.

    A pair is an ``AIMessage`` with exactly one tool call whose name is in
    *collapse_tools*, immediately followed by a ``ToolMessage``.  Consecutive
    pairs form a group.  Only groups with at least *min_group_size* pairs are
    returned.

    Args:
        messages: Current message list.
        collapse_tools: Tool names eligible for collapsing.
        min_group_size: Minimum consecutive pairs to form a group.

    Returns:
        List of ``(start, end)`` index tuples where *end* is exclusive.
    """
    groups: list[tuple[int, int]] = []
    n = len(messages)
    i = 0

    while i < n - 1:
        msg = messages[i]
        if not (
            isinstance(msg, AIMessage)
            and len(msg.tool_calls) == 1
            and msg.tool_calls[0].get("name") in collapse_tools
            and isinstance(messages[i + 1], ToolMessage)
        ):
            i += 1
            continue

        j = i + 2
        group_count = 1

        while j + 1 < n:
            candidate = messages[j]
            if not (
                isinstance(candidate, AIMessage)
                and len(candidate.tool_calls) == 1
                and candidate.tool_calls[0].get("name") in collapse_tools
                and isinstance(messages[j + 1], ToolMessage)
            ):
                break
            group_count += 1
            j += 2

        if group_count >= min_group_size:
            groups.append((i, j))

        i = j

    return groups


def _build_badge(count: int) -> str:
    """Build placeholder text for a collapsed group.

    Args:
        count: Number of tool-call pairs that were collapsed.

    Returns:
        Human-readable badge string.
    """
    return f"[{count} tool results omitted (most recent result preserved below)]"


def _collapse_messages(
    messages: list[AnyMessage],
    collapse_tools: frozenset[str],
    min_group_size: int,
) -> list[AnyMessage]:
    """Collapse consecutive read/search groups, keeping the last pair.

    For each collapsible group the first *N - 1* pairs are replaced with a
    single ``HumanMessage`` badge.  The final pair is kept intact so the
    model retains visibility of the most recent result.

    Args:
        messages: Message list to process.
        collapse_tools: Tool names eligible for collapsing.
        min_group_size: Minimum consecutive pairs to form a group.

    Returns:
        New message list with collapsed groups, or a shallow copy of the
        original when no groups are found.
    """
    groups = _find_collapsible_groups(messages, collapse_tools, min_group_size)

    if not groups:
        return list(messages)

    result: list[AnyMessage] = []
    prev_end = 0
    total_collapsed = 0

    for start, end in groups:
        result.extend(messages[prev_end:start])

        group_size = (end - start) // 2
        collapsed_count = group_size - 1
        total_collapsed += collapsed_count

        if collapsed_count > 0:
            result.append(
                HumanMessage(
                    content=_build_badge(collapsed_count),
                    additional_kwargs={"lc_source": "collapse"},
                ),
            )

        result.append(messages[end - 2])
        result.append(messages[end - 1])

        prev_end = end

    result.extend(messages[prev_end:])

    logger.debug(
        "Collapsed %d groups, removed %d tool-call pairs (%d messages)",
        len(groups),
        total_collapsed,
        total_collapsed * 2,
    )

    return result


class CollapseMiddleware(AgentMiddleware):
    """Collapse consecutive read/search tool calls to reduce context size.

    Scans the message list for consecutive groups of single-tool-call
    ``AIMessage``/``ToolMessage`` pairs where the tool name is in
    *collapse_tools*.  Replaces all but the most recent pair in each group
    with a short placeholder.

    Stateless. Derives groups from ``request.messages`` on every call.

    Args:
        collapse_tools: Tool names eligible for collapsing.  Defaults to
            ``{"read_file", "grep", "glob", "web_search"}``.
        min_group_size: Minimum consecutive pairs required to form a
            collapsible group.  Must be >= 2.  Defaults to 2.

    Raises:
        ValueError: If *min_group_size* is less than 2.

    Example::

        from langchain_collapse import CollapseMiddleware

        middleware = [CollapseMiddleware()]
    """

    state_schema = AgentState

    def __init__(
        self,
        *,
        collapse_tools: frozenset[str] | None = None,
        min_group_size: int = _DEFAULT_MIN_GROUP_SIZE,
    ) -> None:
        """Create a new ``CollapseMiddleware``."""
        if min_group_size < 2:
            msg = f"min_group_size must be >= 2, got {min_group_size}"
            raise ValueError(msg)

        self._collapse_tools = (
            collapse_tools if collapse_tools is not None else _DEFAULT_COLLAPSE_TOOLS
        )
        self._min_group_size = min_group_size

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Collapse consecutive tool-call groups before the model call.

        Args:
            request: Model request to process.
            handler: Handler to call with the modified request.

        Returns:
            Model response from *handler*.
        """
        collapsed = _collapse_messages(
            request.messages,
            self._collapse_tools,
            self._min_group_size,
        )
        return handler(request.override(messages=collapsed))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Collapse consecutive tool-call groups before the model call.

        Args:
            request: Model request to process.
            handler: Async handler to call with the modified request.

        Returns:
            Model response from *handler*.
        """
        collapsed = _collapse_messages(
            request.messages,
            self._collapse_tools,
            self._min_group_size,
        )
        return await handler(request.override(messages=collapsed))
