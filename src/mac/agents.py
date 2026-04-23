"""Agent implementations.

Each agent is an async function `(state) -> partial_state_update`. They are
wired together in orchestrator.py via LangGraph.

Agents communicate exclusively through the RunState — no hidden side channels.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import Config
from .llm import complete, complete_json
from .sandbox import Sandbox
from .state import (
    AgentTurn,
    Review,
    RunState,
    TestCase,
    TestReport,
    TestResult,
)

logger = logging.getLogger(__name__)


def _turn(agent: str, resp: Any, summary: str) -> AgentTurn:
    return AgentTurn(
        agent=agent,
        model=resp.model,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        duration_ms=resp.duration_ms,
        summary=summary,
    )


def _update_budget(state: RunState, resp: Any) -> dict[str, Any]:
    return {"tokens_spent": state.tokens_spent + resp.tokens_in + resp.tokens_out}


# ---------------------------------------------------------------------------
# System Analyst
# ---------------------------------------------------------------------------


async def analyst_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("analyst")
    user = (
        f"User task:\n\n{state.task}\n\n"
        f"Target language hint (may be None, meaning you decide): {state.language or 'unspecified'}\n\n"
        "Write a concise specification with these sections in Markdown:\n"
        "1. Problem statement (one paragraph)\n"
        "2. Inputs (types, bounds)\n"
        "3. Outputs (types, format)\n"
        "4. Constraints and invariants\n"
        "5. Edge cases worth highlighting\n"
        "6. Recommended target language (one line, pick one if not specified; bias toward Python unless the task demands otherwise)"
    )
    resp = await complete(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
    )
    spec = resp.text.strip()
    language = state.language
    if not language:
        language = _extract_language(spec) or "Python"

    return {
        "spec": spec,
        "language": language,
        "history": [
            *state.history,
            _turn("analyst", resp, f"spec len={len(spec)}; lang={language}"),
        ],
        **_update_budget(state, resp),
    }


def _extract_language(spec: str) -> str | None:
    lowered = spec.lower()
    # Crude but effective: look for "recommended target language:" line.
    for line in spec.splitlines():
        if "target language" in line.lower():
            # trim everything before the colon
            tail = line.split(":", 1)[-1].strip().strip(".*_ ")
            if tail:
                return tail.split()[0]
    for lang in ("python", "rust", "c++", "typescript", "go", "java", "c#"):
        if lang in lowered:
            return lang.capitalize() if lang != "c++" else "C++"
    return None


# ---------------------------------------------------------------------------
# Algorithm Designer
# ---------------------------------------------------------------------------


async def designer_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("designer")
    user = (
        f"Specification:\n\n{state.spec}\n\n"
        f"Target language: {state.language}\n\n"
        "Produce an algorithm plan with these sections in Markdown:\n"
        "1. High-level approach (one paragraph)\n"
        "2. Data structures\n"
        "3. Step-by-step algorithm (pseudocode is fine)\n"
        "4. Complexity (time + space) and why it's appropriate\n"
        "5. Alternatives considered and why rejected (briefly)"
    )
    resp = await complete(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
    )
    plan = resp.text.strip()
    return {
        "plan": plan,
        "history": [*state.history, _turn("designer", resp, f"plan len={len(plan)}")],
        **_update_budget(state, resp),
    }


# ---------------------------------------------------------------------------
# Adversarial Test Designer
# ---------------------------------------------------------------------------


async def adversary_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("adversary")
    user = (
        f"Specification:\n\n{state.spec}\n\n"
        f"Plan:\n\n{state.plan}\n\n"
        f"Target language: {state.language}\n\n"
        "Generate 5 to 10 hostile test cases designed to break a naive implementation. "
        "Favor edge cases: empty input, min/max bounds, duplicates, negative numbers, "
        "unicode, off-by-one, numeric overflow, adversarial orderings.\n\n"
        "Return a JSON array where each element is:\n"
        '{"name": str, "description": str, "stdin": str, "expected_stdout": str | null, "kind": "adversarial"}\n\n'
        "`stdin` is what will be piped to the program; `expected_stdout` is what the "
        "correct program must print on stdout (or null if the test only needs to not crash)."
    )
    data, resp = await complete_json(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
        schema_hint='[{"name": str, "description": str, "stdin": str, "expected_stdout": str|null, "kind": "adversarial"}]',
    )
    if isinstance(data, dict) and "tests" in data:
        data = data["tests"]
    if not isinstance(data, list):
        raise ValueError(f"adversary returned non-list: {type(data)}")

    tests = [TestCase(**{**t, "kind": "adversarial"}) for t in data]
    return {
        "adversarial_tests": tests,
        "history": [*state.history, _turn("adversary", resp, f"{len(tests)} adversarial tests")],
        **_update_budget(state, resp),
    }


# ---------------------------------------------------------------------------
# Programmer
# ---------------------------------------------------------------------------

PROGRAMMER_OUTPUT_SCHEMA = (
    '{"files": [{"path": str, "content": str}], '
    '"build_command": str | null, "run_command": str, '
    '"explanation": str}'
)


async def programmer_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("programmer")

    feedback = ""
    if state.review and not state.review.approved:
        feedback = (
            "\n\nPrevious attempt was rejected by the Reviewer with the following "
            "comments. Address every comment.\n"
            f"Verdict: {state.review.verdict}\n"
            + "\n".join(f"- {c}" for c in state.review.comments)
        )
    if state.test_report and not state.test_report.all_passed:
        failed = [r for r in state.test_report.results if not r.passed]
        feedback += "\n\nFailing tests from previous attempt:\n"
        for r in failed[:10]:
            feedback += f"- {r.name}: exit={r.exit_code} stderr={r.stderr[:300]}\n"
    if state.language_notes:
        feedback += f"\n\nLanguage-specialist notes:\n{state.language_notes}"

    user = (
        f"Specification:\n\n{state.spec}\n\n"
        f"Algorithm plan:\n\n{state.plan}\n\n"
        f"Target language: {state.language}\n"
        f"Iteration: {state.iteration}\n"
        f"{feedback}\n\n"
        "Produce a complete, self-contained implementation. Read input from stdin, "
        "write the answer to stdout — this is how the Tester will run it.\n\n"
        "Return JSON with this shape (and nothing else):\n"
        f"{PROGRAMMER_OUTPUT_SCHEMA}\n\n"
        "Rules:\n"
        "- `files[].path` is relative to the project root. Use idiomatic layout for the language.\n"
        "- `build_command` is a single shell command, or null if none is needed (e.g. Python).\n"
        "- `run_command` is a single shell command that runs the program reading stdin.\n"
        "- Do NOT include test harnesses; the Tester supplies stdin separately."
    )

    data, resp = await complete_json(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
        schema_hint=PROGRAMMER_OUTPUT_SCHEMA,
    )
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError(f"programmer returned bad shape: {json.dumps(data)[:500]}")

    files = {f["path"]: f["content"] for f in data["files"]}
    build_command = data.get("build_command") or None
    run_command = data["run_command"]

    return {
        "files": files,
        "build_command": build_command,
        "run_command": run_command,
        # clear previous run artifacts when a new attempt is made
        "test_report": None,
        "review": None,
        "language_notes": None,
        "iteration": state.iteration + 1,
        "history": [
            *state.history,
            _turn("programmer", resp, f"wrote {len(files)} file(s); run=`{run_command}`"),
        ],
        **_update_budget(state, resp),
    }


# ---------------------------------------------------------------------------
# Language Specialist
# ---------------------------------------------------------------------------


async def language_specialist_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("language_specialist")
    files_blob = "\n\n".join(f"### `{p}`\n```\n{c}\n```" for p, c in state.files.items())
    user = (
        f"You are an expert in {state.language}. Review the code below purely from "
        "a language-idiom and correctness perspective. Do NOT rewrite — produce a "
        "short bulleted list of concrete changes the Programmer should make, or the "
        "single word 'OK' (with no bullets) if the code is already idiomatic and sound.\n\n"
        f"Target language: {state.language}\n\n"
        f"Files:\n\n{files_blob}"
    )
    resp = await complete(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
    )
    notes = resp.text.strip()
    notes_out = None if notes.upper().startswith("OK") else notes

    return {
        "language_notes": notes_out,
        "history": [
            *state.history,
            _turn(
                "language_specialist",
                resp,
                "no notes" if notes_out is None else f"notes len={len(notes_out)}",
            ),
        ],
        **_update_budget(state, resp),
    }


# ---------------------------------------------------------------------------
# Verification Engineer (Tester)
# ---------------------------------------------------------------------------


async def tester_node(state: RunState, cfg: Config) -> dict[str, Any]:
    if not state.files or not state.run_command:
        raise RuntimeError("tester_node called without files/run_command")

    sandbox = Sandbox(timeout=cfg.budgets.sandbox_timeout_seconds)
    stdins = [t.stdin for t in state.adversarial_tests]

    if stdins:
        build, runs = await sandbox.run_many(
            state.files,
            build_command=state.build_command,
            run_command=state.run_command,
            stdins=stdins,
        )
    else:
        build, single = await sandbox.run(
            state.files,
            build_command=state.build_command,
            run_command=state.run_command,
        )
        runs = [single]

    build_ok = True
    build_stdout = ""
    build_stderr = ""
    if build is not None:
        build_ok = build.exit_code == 0
        build_stdout = build.stdout
        build_stderr = build.stderr

    results: list[TestResult] = []
    if not build_ok:
        results.append(
            TestResult(
                name="build",
                passed=False,
                stdout=build_stdout,
                stderr=build_stderr,
                exit_code=build.exit_code if build else 1,
                duration_ms=build.duration_ms if build else 0,
            )
        )
    else:
        test_iter = state.adversarial_tests or [
            TestCase(name="default", description="default run with empty stdin")
        ]
        for tc, r in zip(test_iter, runs, strict=False):
            passed = r.exit_code == 0 and not r.timed_out
            if tc.expected_stdout is not None:
                passed = passed and r.stdout.strip() == tc.expected_stdout.strip()
            results.append(
                TestResult(
                    name=tc.name,
                    passed=passed,
                    stdout=r.stdout,
                    stderr=r.stderr,
                    exit_code=r.exit_code,
                    duration_ms=r.duration_ms,
                )
            )

    passed_ct = sum(1 for r in results if r.passed)
    summary = f"{passed_ct}/{len(results)} passed" + ("" if build_ok else " (build failed)")
    report = TestReport(
        results=results,
        build_stdout=build_stdout,
        build_stderr=build_stderr,
        build_ok=build_ok,
        summary=summary,
    )

    # No LLM call here; synthesize a history entry manually.
    return {
        "test_report": report,
        "history": [*state.history, AgentTurn(agent="tester", model="sandbox", summary=summary)],
    }


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------


async def reviewer_node(state: RunState, cfg: Config) -> dict[str, Any]:
    agent = cfg.agent("reviewer")
    files_blob = "\n\n".join(f"### `{p}`\n```\n{c}\n```" for p, c in state.files.items())
    report = state.test_report
    report_blob = ""
    if report:
        report_blob = f"build_ok={report.build_ok}\nsummary={report.summary}\n\n" + "\n".join(
            f"- {r.name}: {'PASS' if r.passed else 'FAIL'} (exit={r.exit_code})"
            + ("" if r.passed else f"\n    stderr: {r.stderr[:300]}")
            for r in report.results
        )

    user = (
        f"Specification:\n\n{state.spec}\n\n"
        f"Algorithm plan:\n\n{state.plan}\n\n"
        f"Target language: {state.language}\n\n"
        f"Code:\n\n{files_blob}\n\n"
        f"Test report:\n{report_blob}\n\n"
        "Review the code and decide whether to approve. Approve ONLY if: all tests "
        "pass, the code matches the spec, and there are no obvious correctness or "
        "idiom issues. Otherwise, reject with concrete, actionable comments.\n\n"
        'Return JSON: {"approved": bool, "verdict": str, "comments": [str]}'
    )
    data, resp = await complete_json(
        cfg,
        agent,
        [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user}],
        schema_hint='{"approved": bool, "verdict": str, "comments": [str]}',
    )
    if not isinstance(data, dict):
        raise ValueError(f"reviewer returned non-object: {data}")

    review = Review(
        approved=bool(data.get("approved", False)),
        verdict=str(data.get("verdict", "")),
        comments=[str(c) for c in data.get("comments", [])],
    )
    return {
        "review": review,
        "history": [
            *state.history,
            _turn(
                "reviewer", resp, f"approved={review.approved}; {len(review.comments)} comment(s)"
            ),
        ],
        **_update_budget(state, resp),
    }
