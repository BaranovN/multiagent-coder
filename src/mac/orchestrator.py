"""LangGraph wiring of the agent team.

The "Orchestrator" is the graph itself — conditional edges decide the next
agent based on state. This avoids spending tokens on a routing LLM call per
hop.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]

from .agents import (
    adversary_node,
    analyst_node,
    designer_node,
    language_specialist_node,
    programmer_node,
    reviewer_node,
    tester_node,
)
from .config import Config, load_config
from .state import RunState

logger = logging.getLogger(__name__)


def build_graph(cfg: Config | None = None):
    cfg = cfg or load_config()

    async def _analyst(s: RunState) -> dict:
        return await analyst_node(s, cfg)

    async def _designer(s: RunState) -> dict:
        return await designer_node(s, cfg)

    async def _adversary(s: RunState) -> dict:
        return await adversary_node(s, cfg)

    async def _programmer(s: RunState) -> dict:
        return await programmer_node(s, cfg)

    async def _language_specialist(s: RunState) -> dict:
        return await language_specialist_node(s, cfg)

    async def _tester(s: RunState) -> dict:
        return await tester_node(s, cfg)

    async def _reviewer(s: RunState) -> dict:
        return await reviewer_node(s, cfg)

    def after_reviewer(s: RunState) -> Literal["end", "programmer"]:
        if s.review and s.review.approved:
            return "end"
        if s.iteration >= cfg.budgets.max_iterations:
            return "end"
        if s.tokens_spent >= cfg.budgets.max_tokens_per_run:
            return "end"
        return "programmer"

    g: StateGraph = StateGraph(RunState)
    g.add_node("analyst", _analyst)
    g.add_node("designer", _designer)
    g.add_node("adversary", _adversary)
    g.add_node("programmer", _programmer)
    g.add_node("language_specialist", _language_specialist)
    g.add_node("tester", _tester)
    g.add_node("reviewer", _reviewer)

    g.add_edge(START, "analyst")
    g.add_edge("analyst", "designer")
    g.add_edge("designer", "adversary")
    g.add_edge("adversary", "programmer")
    g.add_edge("programmer", "language_specialist")
    # Language specialist's notes become feedback for the NEXT Programmer
    # iteration (if any) — they don't loop back immediately, to keep the
    # graph bounded and avoid chatty-specialist deadlocks.
    g.add_edge("language_specialist", "tester")
    g.add_edge("tester", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        after_reviewer,
        {"programmer": "programmer", "end": END},
    )

    return g.compile()


async def run_task(task: str, *, language: str | None = None) -> RunState:
    cfg = load_config()
    graph = build_graph(cfg)
    state = RunState(task=task, language=language)
    # Each straight-through pass is: programmer -> language_specialist ->
    # tester -> reviewer -> (loop). Four nodes per iteration + the initial
    # analyst/designer/adversary head, plus slack for conditional hops.
    recursion_limit = max(40, cfg.budgets.max_iterations * 5 + 15)
    final = await graph.ainvoke(state, config={"recursion_limit": recursion_limit})

    # LangGraph returns either a RunState instance or a dict depending on version.
    if isinstance(final, RunState):
        return final
    return RunState.model_validate(final)
