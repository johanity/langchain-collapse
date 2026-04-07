"""Property-based invariant tests.

Verify that structural properties hold for any valid input, not just
hand-picked examples.  Four invariants are tested:

1. Output never exceeds input length.
2. Last ToolMessage in every collapsed group survives.
3. Every ToolMessage in the output has a matching AIMessage.
4. Collapsing is idempotent.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from langchain_collapse import _collapse_messages
from tests.conftest import make_read_pairs

_TOOLS: frozenset[str] = frozenset({"read_file", "grep", "glob", "web_search"})


class TestNeverGrows:
    """Output must never contain more messages than input."""

    @given(n=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_consecutive_reads(self, n: int) -> None:
        """N consecutive reads never produce more messages than 2N."""
        msgs = make_read_pairs(n)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        assert len(result) <= len(msgs)

    @given(n=st.integers(min_value=0, max_value=50))
    @settings(max_examples=30)
    def test_reads_with_human_prefix(self, n: int) -> None:
        """A leading HumanMessage does not break the invariant."""
        msgs: list[AnyMessage] = [HumanMessage(content="start"), *make_read_pairs(n)]
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        assert len(result) <= len(msgs)


class TestLastResultPreserved:
    """Final ToolMessage in a collapsed group must survive."""

    @given(n=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_last_content_survives(self, n: int) -> None:
        """The last file's content is always present in the output."""
        msgs = make_read_pairs(n)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        last_content = f"content of file {n - 1}"
        tool_contents = [m.content for m in result if isinstance(m, ToolMessage)]
        assert last_content in tool_contents


class TestNoOrphanedResults:
    """Every ToolMessage in the output must have a matching AIMessage."""

    @given(n=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_tool_ids_subset_of_ai_ids(self, n: int) -> None:
        """ToolMessage.tool_call_id is a subset of AIMessage tool-call IDs."""
        msgs = make_read_pairs(n)
        result = _collapse_messages(msgs, _TOOLS, min_group_size=2)

        ai_ids: set[str] = set()
        tool_ids: set[str] = set()
        for msg in result:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls:
                    ai_ids.add(tc["id"])
            elif isinstance(msg, ToolMessage):
                tool_ids.add(msg.tool_call_id)

        assert tool_ids <= ai_ids


class TestIdempotent:
    """Collapsing the output again must produce the same result."""

    @given(n=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_double_collapse_is_noop(self, n: int) -> None:
        """collapse(collapse(msgs)) == collapse(msgs)."""
        msgs = make_read_pairs(n)
        once = _collapse_messages(msgs, _TOOLS, min_group_size=2)
        twice = _collapse_messages(once, _TOOLS, min_group_size=2)
        assert len(once) == len(twice)
        for a, b in zip(once, twice, strict=True):
            assert type(a) is type(b)
            assert a.content == b.content
