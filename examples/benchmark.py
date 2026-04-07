"""Measure token savings on a realistic agent session.

Simulates a coding agent that reads files, searches for patterns, makes
edits, then reads more files.  Shows how collapsing affects the point at
which SummarizationMiddleware would trigger.

Usage::

    python examples/benchmark.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from langchain_collapse import _collapse_messages

_TOOLS: frozenset[str] = frozenset({"read_file", "grep", "glob", "web_search"})

_FILE_CONTENT = """\
from flask import Blueprint, request, jsonify
from auth import require_auth, verify_token
from models import User, Session
from db import get_connection

api = Blueprint("api", __name__)

@api.route("/users", methods=["GET"])
@require_auth
def list_users():
    conn = get_connection()
    users = conn.execute("SELECT * FROM users WHERE active = 1").fetchall()
    return jsonify([dict(u) for u in users])

@api.route("/users/<int:user_id>", methods=["DELETE"])
@require_auth
def delete_user(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return jsonify({"status": "deleted"})
"""


def _build_session() -> list[AnyMessage]:
    """Build a realistic agent session with reads, searches, edits, re-reads."""
    msgs: list[AnyMessage] = [HumanMessage(content="Fix the auth bug in the API")]
    cid = 0

    for name in [
        "auth.py",
        "models.py",
        "db.py",
        "routes.py",
        "config.py",
        "tests/test_auth.py",
        "tests/test_routes.py",
        "utils.py",
    ]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": f"/src/{name}"},
                        "id": f"tc{cid}",
                    }
                ],
            )
        )
        msgs.append(ToolMessage(content=_FILE_CONTENT, tool_call_id=f"tc{cid}"))

    for pattern in ["verify_token", "session_expired", "401", "unauthorized"]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "grep", "args": {"pattern": pattern}, "id": f"tc{cid}"}
                ],
            )
        )
        msgs.append(
            ToolMessage(
                content=(
                    f"src/auth.py:42: if {pattern}:\n"
                    f"  src/routes.py:18: # {pattern} check"
                ),
                tool_call_id=f"tc{cid}",
            )
        )

    cid += 1
    msgs.append(
        AIMessage(
            content="Found the bug. Fixing now.",
            tool_calls=[
                {
                    "name": "edit_file",
                    "args": {"path": "/src/auth.py"},
                    "id": f"tc{cid}",
                }
            ],
        )
    )
    msgs.append(
        ToolMessage(content="File edited successfully.", tool_call_id=f"tc{cid}")
    )

    for name in ["auth.py", "routes.py", "tests/test_auth.py"]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": f"/src/{name}"},
                        "id": f"tc{cid}",
                    }
                ],
            )
        )
        msgs.append(ToolMessage(content=_FILE_CONTENT, tool_call_id=f"tc{cid}"))

    for pattern in ["verify_token", "session_expired"]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "grep", "args": {"pattern": pattern}, "id": f"tc{cid}"}
                ],
            )
        )
        msgs.append(
            ToolMessage(
                content=f"src/auth.py:42: if {pattern}: # fixed",
                tool_call_id=f"tc{cid}",
            )
        )

    return msgs


def _summarization_trigger_point(
    token_budget: int,
    tokens_per_pair: int,
    *,
    collapse: bool,
) -> int:
    """Estimate how many tool-call pairs fit before hitting a token budget.

    With collapsing, each group of N pairs compresses to 1 badge + 1 pair,
    so the effective cost per pair drops significantly for consecutive
    read/search operations.
    """
    if not collapse:
        return token_budget // tokens_per_pair

    badge_tokens = 25
    pair_tokens = tokens_per_pair
    # Worst case: groups of 5 pairs → badge(25) + last pair(pair_tokens)
    # = (25 + pair_tokens) tokens for 5 pairs of work
    effective_per_pair = (badge_tokens + pair_tokens) / 5
    return int(token_budget / effective_per_pair)


def main() -> None:
    """Run the benchmark."""
    msgs = _build_session()
    collapsed = _collapse_messages(msgs, _TOOLS, min_group_size=2)

    before_tokens = count_tokens_approximately(msgs)
    after_tokens = count_tokens_approximately(collapsed)

    print("langchain-collapse benchmark")
    print("=" * 50)
    print(f"{'':>28} {'Before':>8} {'After':>8}")
    print(f"{'Messages':>28} {len(msgs):>8} {len(collapsed):>8}")
    print(f"{'Approx tokens':>28} {before_tokens:>8} {after_tokens:>8}")
    reduction = (1 - after_tokens / before_tokens) * 100
    print(f"{'Token reduction':>28} {'':>8} {reduction:>7.0f}%")
    print()

    # Show when SummarizationMiddleware would trigger.
    budget = 170_000  # 85% of 200K context window
    tokens_per_pair = before_tokens // (len(msgs) // 2)
    without = _summarization_trigger_point(budget, tokens_per_pair, collapse=False)
    with_ = _summarization_trigger_point(budget, tokens_per_pair, collapse=True)

    print("SummarizationMiddleware trigger point (at 85% of 200K):")
    print(f"  Without CollapseMiddleware:  ~{without} tool calls")
    print(f"  With CollapseMiddleware:     ~{with_} tool calls")
    ratio = with_ / without
    print(f"  Ratio:                       {ratio:.1f}x more work before summarization")


if __name__ == "__main__":
    main()
