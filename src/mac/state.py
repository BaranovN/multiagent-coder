"""Shared run state passed between agents.

Every agent reads this state and returns a partial dict of updates that
LangGraph merges in. Keep it small and JSON-serializable so it can be
persisted, streamed, and traced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    name: str
    description: str
    stdin: str = ""
    expected_stdout: str | None = None
    kind: Literal["unit", "adversarial"] = "unit"


class TestResult(BaseModel):
    name: str
    passed: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0


class TestReport(BaseModel):
    results: list[TestResult] = Field(default_factory=list)
    build_stdout: str = ""
    build_stderr: str = ""
    build_ok: bool = True
    summary: str = ""

    @property
    def all_passed(self) -> bool:
        return self.build_ok and all(r.passed for r in self.results)


class Review(BaseModel):
    approved: bool
    verdict: str
    comments: list[str] = Field(default_factory=list)


class AgentTurn(BaseModel):
    agent: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    summary: str = ""


class RunState(BaseModel):
    """The single source of truth shared across all agents."""

    task: str
    language: str | None = None

    spec: str | None = None
    plan: str | None = None
    adversarial_tests: list[TestCase] = Field(default_factory=list)

    # filename -> file contents
    files: dict[str, str] = Field(default_factory=dict)
    build_command: str | None = None
    run_command: str | None = None

    language_notes: str | None = None
    test_report: TestReport | None = None
    review: Review | None = None

    iteration: int = 0
    tokens_spent: int = 0
    history: list[AgentTurn] = Field(default_factory=list)

    # When the graph terminates because a budget was exhausted rather than
    # because the reviewer approved, we still want to surface the best effort.
    done_reason: str | None = None
