"""Loads agent prompts, model routing, and budgets from YAML + env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    model: str  # litellm model string, e.g. "groq/llama-3.3-70b-versatile"
    api_key_env: str | None = None  # name of env var that MUST be set for this model
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    name: str
    model: str  # key into ModelSpec table
    fallback: list[str] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 4096
    system_prompt: str


class Budgets(BaseModel):
    max_iterations: int = 4
    max_tokens_per_run: int = 200_000
    agent_timeout_seconds: int = 120
    sandbox_timeout_seconds: int = 30


class Config(BaseModel):
    models: dict[str, ModelSpec]
    agents: dict[str, AgentSpec]
    budgets: Budgets

    def agent(self, name: str) -> AgentSpec:
        if name not in self.agents:
            raise KeyError(f"Unknown agent: {name}. Known: {list(self.agents)}")
        return self.agents[name]

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(f"Unknown model key: {key}. Known: {list(self.models)}")
        return self.models[key]


DEFAULT_CONFIG_PATH = Path(__file__).parent / "prompts" / "agents.yaml"


def _apply_env_overrides(cfg: Config) -> Config:
    """Allow overriding a few knobs via environment variables."""
    if v := os.getenv("MAC_MAX_ITERATIONS"):
        cfg.budgets.max_iterations = int(v)
    if v := os.getenv("MAC_MAX_TOKENS_PER_RUN"):
        cfg.budgets.max_tokens_per_run = int(v)
    if v := os.getenv("MAC_SANDBOX_TIMEOUT"):
        cfg.budgets.sandbox_timeout_seconds = int(v)
    return cfg


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config. Cached per path."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = yaml.safe_load(p.read_text())
    cfg = Config.model_validate(data)
    return _apply_env_overrides(cfg)


def available_models(cfg: Config) -> list[str]:
    """Return model keys that have their API key set (so we can actually call them)."""
    out = []
    for key, spec in cfg.models.items():
        if spec.api_key_env is None or os.getenv(spec.api_key_env):
            out.append(key)
    return out
