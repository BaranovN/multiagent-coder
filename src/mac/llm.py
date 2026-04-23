"""Thin wrapper over LiteLLM with provider fallback and structured-output helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from .config import AgentSpec, Config, ModelSpec

logger = logging.getLogger(__name__)

# LiteLLM spams by default; we only want our own logs.
litellm.suppress_debug_info = True
litellm.drop_params = True  # silently drop params a provider doesn't understand


RETRIABLE = (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError, APIError)


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    raw: Any


def _model_available(spec: ModelSpec) -> bool:
    return spec.api_key_env is None or bool(os.getenv(spec.api_key_env))


async def _call_once(
    spec: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
    timeout: int,
) -> LLMResponse:
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    kwargs.update(spec.extra)
    if response_format is not None:
        kwargs["response_format"] = response_format

    t0 = time.monotonic()
    resp = await litellm.acompletion(**kwargs)
    dur_ms = int((time.monotonic() - t0) * 1000)

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
    tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

    return LLMResponse(
        text=text,
        model=spec.model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=dur_ms,
        raw=resp,
    )


async def complete(
    cfg: Config,
    agent: AgentSpec,
    messages: list[dict[str, str]],
    *,
    response_format: dict[str, Any] | None = None,
    max_attempts: int = 2,
) -> LLMResponse:
    """Call the agent's primary model, falling back through its declared chain.

    Retries each model once on retriable errors, then moves to the next model.
    If every model fails, raises the last error.
    """
    timeout = cfg.budgets.agent_timeout_seconds
    candidates: list[str] = [agent.model, *agent.fallback]

    last_exc: Exception | None = None
    for key in candidates:
        spec = cfg.model(key)
        if not _model_available(spec):
            logger.info("skip model %s: %s not set", key, spec.api_key_env)
            continue

        for attempt in range(max_attempts):
            try:
                return await _call_once(
                    spec,
                    messages,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens,
                    response_format=response_format,
                    timeout=timeout,
                )
            except RETRIABLE as e:
                last_exc = e
                wait = 1.5 * (attempt + 1)
                logger.warning(
                    "model %s attempt %d failed (%s); retrying in %.1fs",
                    spec.model,
                    attempt + 1,
                    type(e).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as e:
                # Non-retriable error for this model; move on to the next.
                last_exc = e
                logger.warning("model %s non-retriable error: %s", spec.model, e)
                break

    if last_exc:
        raise last_exc
    raise RuntimeError(
        f"No usable models for agent '{agent.name}'. "
        f"Set one of the API keys required by: {candidates}"
    )


async def complete_json(
    cfg: Config,
    agent: AgentSpec,
    messages: list[dict[str, str]],
    *,
    schema_hint: str | None = None,
) -> tuple[dict[str, Any] | list[Any], LLMResponse]:
    """Ask the model for JSON output and parse it.

    Strategy: always append a strict "JSON only" instruction to the prompt.
    Try native JSON mode on the primary provider only (so a single
    provider's broken JSON mode doesn't burn through all fallbacks). If
    that fails, fall back to plain-mode completion with the full provider
    chain.
    """
    plain_messages = list(messages)
    instruction = (
        "Respond with a single valid JSON value and nothing else. "
        "Do not wrap it in markdown fences or add commentary."
    )
    if schema_hint:
        instruction += f" The JSON must match this schema: {schema_hint}"
    plain_messages.append({"role": "system", "content": instruction})

    primary = cfg.model(agent.model)
    if _model_available(primary):
        try:
            resp = await _call_once(
                primary,
                plain_messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                response_format={"type": "json_object"},
                timeout=cfg.budgets.agent_timeout_seconds,
            )
            return _parse_json_loose(resp.text), resp
        except Exception as e:
            logger.info(
                "native JSON mode on %s failed (%s); falling back to plain chain",
                primary.model,
                type(e).__name__,
            )

    resp = await complete(cfg, agent, plain_messages)
    return _parse_json_loose(resp.text), resp


def _parse_json_loose(text: str) -> dict[str, Any] | list[Any]:
    """Try hard to extract JSON from a model's answer."""
    t = text.strip()
    # Strip markdown fences if present.
    if t.startswith("```"):
        t = t.strip("`")
        # drop optional language tag on first line
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Last-ditch: find the outermost JSON object/array.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i = t.find(open_c)
        j = t.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:2000]}")
