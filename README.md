# multiagent-coder

A small, pragmatic multi-agent system for solving coding tasks. An **Orchestrator** routes a task through a team of specialists who each see a shared scratchpad and produce artifacts:

```
                     ┌────────────────┐
                     │  Orchestrator  │
                     └───────┬────────┘
      ┌───────┬───────┬──────┼──────┬───────────────┬──────────┐
      ▼       ▼       ▼      ▼      ▼               ▼          ▼
 ┌────────┐ ┌──────┐ ┌─────┐ ┌────────┐ ┌────────────────┐ ┌────────┐
 │ System │ │ Algo │ │ Adv │ │  Prog  │ │ X‑Lang Special │ │ Tester │
 │Analyst │ │Design│ │Tests│ │rammer  │ │                │ │        │
 └────────┘ └──────┘ └─────┘ └────────┘ └────────────────┘ └────────┘
                              │
                              ▼
                         ┌─────────┐
                         │Reviewer │
                         └─────────┘
```

- **Orchestrator** — decides what to do next based on the shared state. Picks the language, routes between specialists, enforces budget.
- **System Analyst** — turns the user task into a concrete spec: inputs, outputs, invariants, constraints.
- **Algorithm Designer** — proposes algorithms, data structures, complexity.
- **Adversarial Test Designer** — generates hostile tests (empty inputs, max sizes, off-by-one, UB).
- **Programmer** — writes the actual code, iterates using tools (`write_file`, `run_shell`, `run_tests`).
- **X Language Specialist** — the language expert (Rust/C++/Python/TS…), reviews idiomatic usage.
- **Verification Engineer (Tester)** — executes tests in a subprocess sandbox, reports failures.
- **Reviewer** — final gate: reads code + test report, either approves or sends back with comments.

## Stack

| Layer            | Choice                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------- |
| Orchestration    | [LangGraph](https://langchain-ai.github.io/langgraph/)                                  |
| LLM routing      | [LiteLLM](https://docs.litellm.ai) — one API for Groq, Gemini, OpenAI, Anthropic, Ollama |
| Models (default) | Groq `llama-3.3-70b-versatile` + Gemini `gemini-2.0-flash` (both free tier)             |
| Local fallback   | Ollama (`qwen2.5-coder:7b` recommended)                                                 |
| Sandbox          | Subprocess with timeouts + ephemeral workdir. Optional Docker sandbox.                   |
| Tracing          | Optional [Langfuse](https://langfuse.com) (self-host via `docker-compose up`)           |
| Package mgr      | [uv](https://docs.astral.sh/uv/)                                                        |

## Quick start

```bash
# 1. Install
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync

# 2. Configure
cp .env.example .env
# Edit .env and put in at least one of GROQ_API_KEY or GEMINI_API_KEY.
# Get them free at:
#   https://console.groq.com/keys
#   https://aistudio.google.com/apikey

# 3. Run the demo task (two-sum)
uv run mac solve examples/two_sum_task.md

# Or give your own task inline:
uv run mac solve --task "Write a function that returns all primes up to N using a sieve of Eratosthenes. Target language: Rust."
```

Output artifacts (code, tests, run logs) land in `runs/<timestamp>/`.

## Architecture

### State

All agents read and write a single pydantic `RunState` (see `src/mac/state.py`). Fields:

- `task` — the raw user task.
- `spec` — populated by the Analyst.
- `plan` — populated by the Algorithm Designer.
- `adversarial_tests` — populated by the Adversarial Test Designer.
- `language` — chosen by Orchestrator (or explicit in the task).
- `files` — mapping of relative path → content (the code Programmer writes).
- `test_report` — populated by Tester.
- `review` — populated by Reviewer. If `approved=False`, loop continues.
- `history` — append-only log of agent turns for debugging/tracing.
- `iteration` / `tokens_spent` — budget tracking.

### Graph

```
START → analyst → designer → adversary → programmer ⇄ language_specialist
                                              │
                                              ▼
                                           tester
                                              │
                                              ▼
                                          reviewer
                                          │      │
                                    approved    needs_fix
                                          │      │
                                          ▼      └──► programmer (loop)
                                         END
```

The Orchestrator is modeled as the conditional-edge logic inside LangGraph (not a separate LLM call per hop) — this is faster and much cheaper than a talking router, and matches "supervisor is state-machine, workers are LLM calls".

### Budgets

Every run enforces:

- `max_iterations` — Reviewer → Programmer loop cap.
- `max_tokens_per_run` — aggregate token budget.
- Per-agent timeout (configurable in `src/mac/prompts/agents.yaml`).

If any budget is exhausted, the run ends gracefully with the best artifacts produced so far.

## Development

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Roadmap

- [ ] Docker-based sandbox (currently subprocess).
- [ ] Postgres + pgvector memory across runs.
- [ ] MCP tool server so agents can be reused from other orchestrators.
- [ ] Web UI to watch agents in real time.
- [ ] Self-play evaluation on a standard benchmark (HumanEval / LiveCodeBench).

## License

MIT.
