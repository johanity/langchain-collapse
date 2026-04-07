"""Unit tests for group detection, message transformation, and middleware."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from langchain_collapse import (
    CollapseMiddleware,
    _build_badge,
    _collapse_messages,
    _find_collapsible_groups,
)
from tests.conftest import make_ai, make_read_pairs, make_tool

_TOOLS: frozenset[str] = frozenset({"read_file", "grep", "glob", "web_search"})


class TestFindCollapsibleGroups:
    """Group detection algorithm."""

    def test_finds_consecutive_pairs(self) -> None:
        """Four consecutive read_file pairs form one group."""
        msgs = make_read_pairs(4)
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == [(0, 8)]

    def test_skips_non_collapsible_tools(self) -> None:
        """Tool calls outside the collapse set are ignored."""
        msgs = [
            make_ai("write_file", "tc0"),
            make_tool("tc0"),
            make_ai("write_file", "tc1"),
            make_tool("tc1"),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == []

    def test_respects_min_group_size(self) -> None:
        """A single pair does not meet min_group_size of 2."""
        msgs = make_read_pairs(1)
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == []

    def test_breaks_on_human_message(self) -> None:
        """A HumanMessage between pairs breaks the group."""
        msgs = [
            *make_read_pairs(1),
            HumanMessage(content="interruption"),
            *make_read_pairs(1),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == []

    def test_breaks_on_multi_tool_call(self) -> None:
        """An AIMessage with multiple tool calls breaks the group."""
        multi = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {}, "id": "m0"},
                {"name": "grep", "args": {}, "id": "m1"},
            ],
        )
        msgs = [
            *make_read_pairs(1),
            multi,
            make_tool("m0"),
            make_tool("m1"),
            *make_read_pairs(1),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == []

    def test_handles_empty_messages(self) -> None:
        """Empty message list returns no groups."""
        groups = _find_collapsible_groups([], _TOOLS, min_group_size=2)
        assert groups == []

    def test_handles_mixed_collapsible_tools(self) -> None:
        """Different collapsible tools in sequence form one group."""
        msgs: list[AnyMessage] = [
            make_ai("read_file", "tc0"),
            make_tool("tc0"),
            make_ai("grep", "tc1"),
            make_tool("tc1"),
            make_ai("glob", "tc2"),
            make_tool("tc2"),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == [(0, 6)]

    def test_multiple_separate_groups(self) -> None:
        """Non-collapsible tool between two groups yields two groups."""
        msgs: list[AnyMessage] = [
            *make_read_pairs(2),
            make_ai("write_file", "gap"),
            make_tool("gap"),
            *make_read_pairs(3),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert len(groups) == 2
        assert groups[0] == (0, 4)
        assert groups[1] == (6, 12)

    def test_group_at_end_of_messages(self) -> None:
        """A group that extends to the last message is handled correctly."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="start"),
            *make_read_pairs(3),
        ]
        groups = _find_collapsible_groups(msgs, _TOOLS, min_group_size=2)
        assert groups == [(1, 7)]


class TestCollapseMessages:
    """Message transformation logic."""

    def test_collapses_group_preserving_last_pair(self) -> None:
        """Five pairs collapse to badge + last pair (3 messages total)."""
        msgs = make_read_pairs(5)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert "4 tool results omitted" in result[0].content
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], ToolMessage)
        assert result[2].content == "content of file 4"

    def test_badge_carries_lc_source(self) -> None:
        """Badge HumanMessage is tagged so downstream middleware can identify it."""
        msgs = make_read_pairs(3)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        badge = result[0]
        assert badge.additional_kwargs.get("lc_source") == "collapse"

    def test_returns_original_when_no_groups(self) -> None:
        """Messages without collapsible tools pass through unchanged."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "hi"

    def test_badge_format(self) -> None:
        """Badge string matches the expected format."""
        badge = _build_badge(4)
        expected = "[4 tool results omitted (most recent result preserved below)]"
        assert badge == expected

    def test_multiple_groups(self) -> None:
        """Two separate groups are each collapsed independently."""
        msgs: list[AnyMessage] = [
            *make_read_pairs(3),
            HumanMessage(content="gap"),
            *make_read_pairs(2),
        ]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        badges = [
            m for m in result if isinstance(m, HumanMessage) and "omitted" in m.content
        ]
        assert len(badges) == 2

    def test_preserves_messages_between_groups(self) -> None:
        """Non-group messages between groups are kept intact."""
        msgs: list[AnyMessage] = [
            *make_read_pairs(2),
            HumanMessage(content="keep me"),
            *make_read_pairs(2),
        ]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        contents = [m.content for m in result if isinstance(m, HumanMessage)]
        assert "keep me" in contents

    def test_no_orphaned_tool_call_ids(self) -> None:
        """Every ToolMessage in the output has a matching AIMessage."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="start"),
            *make_read_pairs(5),
            HumanMessage(content="end"),
        ]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        ai_ids: set[str] = set()
        tool_ids: set[str] = set()
        for msg in result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    ai_ids.add(tc["id"])
            elif isinstance(msg, ToolMessage):
                tool_ids.add(msg.tool_call_id)

        assert tool_ids <= ai_ids

    def test_single_pair_not_collapsed(self) -> None:
        """A single pair below min_group_size is left unchanged."""
        msgs = make_read_pairs(1)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        assert len(result) == 2

    def test_coexists_with_summarization_badge(self) -> None:
        """Collapse badges and summarization badges coexist without confusion.

        After SummarizationMiddleware runs, the message list starts with a
        HumanMessage tagged ``lc_source: "summarization"``.  Collapsing must
        not remove or confuse that badge with its own.
        """
        summarization_badge = HumanMessage(
            content="Summary of previous conversation...",
            additional_kwargs={"lc_source": "summarization"},
        )
        msgs: list[AnyMessage] = [
            summarization_badge,
            *make_read_pairs(4),
        ]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        # Summarization badge preserved at position 0.
        assert result[0] is summarization_badge
        assert result[0].additional_kwargs["lc_source"] == "summarization"

        # Collapse badge present with its own lc_source.
        collapse_badges = [
            m
            for m in result
            if isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == "collapse"
        ]
        assert len(collapse_badges) == 1


class TestCollapseMiddleware:
    """Middleware integration surface."""

    def test_wrap_model_call_collapses(self) -> None:
        """Handler receives collapsed messages when groups exist."""
        msgs = make_read_pairs(4)
        request = Mock()
        request.messages = msgs
        override_request = Mock()
        request.override = Mock(return_value=override_request)
        handler = Mock(return_value="response")

        mw = CollapseMiddleware()
        result = mw.wrap_model_call(request, handler)

        request.override.assert_called_once()
        collapsed = request.override.call_args.kwargs["messages"]
        assert len(collapsed) < len(msgs)
        handler.assert_called_once_with(override_request)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_awrap_model_call_collapses(self) -> None:
        """Async handler receives collapsed messages when groups exist."""
        msgs = make_read_pairs(4)
        request = Mock()
        request.messages = msgs
        override_request = Mock()
        request.override = Mock(return_value=override_request)

        async def async_handler(req: object) -> str:
            return "async_response"

        mw = CollapseMiddleware()
        result = await mw.awrap_model_call(request, async_handler)

        request.override.assert_called_once()
        collapsed = request.override.call_args.kwargs["messages"]
        assert len(collapsed) < len(msgs)
        assert result == "async_response"

    def test_custom_collapse_tools(self) -> None:
        """Custom collapse_tools override the defaults."""
        msgs: list[AnyMessage] = [
            make_ai("custom_tool", "tc0"),
            make_tool("tc0"),
            make_ai("custom_tool", "tc1"),
            make_tool("tc1"),
        ]
        mw = CollapseMiddleware(collapse_tools=frozenset({"custom_tool"}))

        result = _collapse_messages(msgs, mw._collapse_tools, mw._min_group_size)
        assert len(result) == 3

    def test_validates_min_group_size(self) -> None:
        """min_group_size below 2 raises ValueError."""
        with pytest.raises(ValueError, match="min_group_size must be >= 2"):
            CollapseMiddleware(min_group_size=1)

    def test_passthrough_when_nothing_to_collapse(self) -> None:
        """Handler gets the original messages when no groups exist."""
        msgs: list[AnyMessage] = [HumanMessage(content="hello")]
        request = Mock()
        request.messages = msgs
        request.override = Mock(return_value=request)
        handler = Mock(return_value="response")

        mw = CollapseMiddleware()
        mw.wrap_model_call(request, handler)

        collapsed = request.override.call_args.kwargs["messages"]
        assert len(collapsed) == 1
        assert collapsed[0].content == "hello"
