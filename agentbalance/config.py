"""Configuration contains profiles and environment-variable names, never API keys."""

from dataclasses import dataclass, field
import json
import math
import os
import re
from pathlib import Path


def number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    model: str
    quality: float
    input_usd_per_million: float
    output_usd_per_million: float
    latency_seconds: float
    endpoint: str = ""
    api_key_env: str = ""
    protocol: str = "chat"
    auth_header: str = "Authorization"
    token_limit_field: str = "max_tokens"
    extra_body: dict = field(default_factory=dict)
    performance_profile: str = ""
    model_type: str = "general"
    calibration_cost_usd: float | None = None

    def cost(self, input_tokens, output_tokens):
        return (input_tokens * self.input_usd_per_million
                + output_tokens * self.output_usd_per_million) / 1_000_000


@dataclass(frozen=True)
class Config:
    models: tuple[Model, ...]
    cheap_model: str
    strong_model: str
    max_output_tokens: int = 256
    context_chars: int = 1200
    request_timeout_seconds: float = 60
    # Ordered templates bound the admissible DAG; the final role is always retained.
    agent_templates: dict[str, str] = field(default_factory=dict)
    min_workers: int = 1
    medium_model: str | None = None
    pool_count: int | None = None
    pool_min_size: int = 1
    pool_max_size: int | None = None
    controller_method: str = "legacy"
    difficulty_offset: float = 0.0
    pool_upper_bound: int | None = None
    explicit_pools: tuple = ()
    decision_model: str | None = None
    scheduler: str = "waves"
    min_request_interval_seconds: float = 0.0
    ablation: str = "none"
    agent_tools: str = "none"
    agent_tool_image: str | None = None
    prompt_prefixes: dict = field(default_factory=dict)
    trace_level: str = "compact"

    def get(self, model_id):
        return next(m for m in self.models if m.id == model_id)


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    models = []
    for item in data["models"]:
        item = dict(item)
        for key in ("input_usd_per_million", "output_usd_per_million"):
            value = item.get(key)
            if isinstance(value, str) and (match := re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)):
                try:
                    item[key] = float(os.environ.get(match[1], ""))
                except ValueError:
                    raise ValueError(f"Set {match[1]} to the numeric price per million tokens for your deployment") from None
        model = Model(**item)
        if not isinstance(model.performance_profile, str) or not isinstance(model.model_type, str):
            raise ValueError("Model performance_profile and model_type must be strings")
        for name in ("id", "model"):
            if not isinstance(getattr(model, name), str) or not getattr(model, name).strip():
                raise ValueError(f"Model {name} must be a nonempty string")
        if model.provider not in {"mock", "groq", "azure", "ollama", "openai-compatible"}:
            raise ValueError(f"Unsupported provider: {model.provider}")
        number(model.quality, "quality", maximum=1)
        if model.calibration_cost_usd is not None:
            number(model.calibration_cost_usd, "calibration_cost_usd")
        for name in ("input_usd_per_million", "output_usd_per_million", "latency_seconds"):
            number(getattr(model, name), f"{model.id}.{name}")
        if model.protocol not in {"chat", "responses"}:
            raise ValueError("protocol must be chat or responses")
        if model.auth_header not in {"Authorization", "api-key"}:
            raise ValueError("auth_header must be Authorization or api-key")
        if model.token_limit_field not in {"max_tokens", "max_completion_tokens", "max_output_tokens"}:
            raise ValueError("Unsupported token_limit_field")
        if model.protocol == "chat" and model.token_limit_field == "max_output_tokens":
            raise ValueError("Chat models require max_tokens or max_completion_tokens")
        if not isinstance(model.extra_body, dict):
            raise ValueError("extra_body must be an object")
        reserved = {"model", "messages", "input", "stream", "store", "max_tokens",
                    "max_completion_tokens", "max_output_tokens", "tools", "n"}
        if reserved.intersection(model.extra_body):
            raise ValueError("extra_body cannot override routing, token caps, streaming, or tools")
        models.append(model)
    if len({m.id for m in models}) != len(models):
        raise ValueError("Model IDs must be unique")
    config = Config(models=tuple(models), **{k: v for k, v in data.items() if k != "models"})
    if config.controller_method not in {"legacy", "paper"}:
        raise ValueError("controller_method must be legacy or paper")
    if config.scheduler not in {"waves", "ready"}:
        raise ValueError("scheduler must be waves or ready")
    if config.trace_level not in {"compact", "full"}:
        raise ValueError("trace_level must be compact or full")
    if config.ablation not in {"none", "random-pool", "random-matching", "role-only", "no-gating", "no-topology"}:
        raise ValueError("Unknown ablation")
    if config.agent_tools not in {"none", "math", "humaneval"}:
        raise ValueError("agent_tools must be none, math, or humaneval")
    if config.agent_tools != "none" and not config.agent_tool_image:
        raise ValueError("Agent tools need agent_tool_image")
    if not isinstance(config.prompt_prefixes, dict) or any(not isinstance(v, str) for v in config.prompt_prefixes.values()):
        raise ValueError("prompt_prefixes must map role IDs to strings")
    number(config.min_request_interval_seconds, "min_request_interval_seconds")
    if config.decision_model is not None and config.decision_model not in {m.id for m in models}:
        raise ValueError("decision_model must name a configured model")
    number(config.difficulty_offset, "difficulty_offset", minimum=-1, maximum=1)
    if config.pool_upper_bound is not None and (isinstance(config.pool_upper_bound, bool)
            or not isinstance(config.pool_upper_bound, int) or config.pool_upper_bound < 0):
        raise ValueError("pool_upper_bound must be a nonnegative, zero-based index")
    if config.explicit_pools:
        ids = {m.id for m in models}
        if any(not p or len(set(p)) != len(p) or not set(p) <= ids for p in config.explicit_pools):
            raise ValueError("explicit_pools must contain nonempty lists of distinct model IDs")
    for name in (config.cheap_model, config.strong_model):
        if name not in {m.id for m in models}:
            raise ValueError(f"Unknown pool model: {name}")
    if config.cheap_model == config.strong_model:
        raise ValueError("Use distinct cheap_model and strong_model profile IDs")
    if config.medium_model is not None and (config.medium_model not in {m.id for m in models}
                                           or config.medium_model in {config.cheap_model, config.strong_model}):
        raise ValueError("medium_model must name a distinct configured model")
    for key in ("pool_count", "pool_min_size"):
        value = getattr(config, key)
        if key == "pool_count" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= len(models):
            raise ValueError(f"{key} must be between one and the number of models")
    if config.pool_max_size is not None and (isinstance(config.pool_max_size, bool)
            or not isinstance(config.pool_max_size, int) or config.pool_max_size < config.pool_min_size):
        raise ValueError("pool_max_size must be an integer at least pool_min_size")
    for name in ("max_output_tokens", "context_chars"):
        value = getattr(config, name)
        number(value, name, minimum=1)
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    number(config.request_timeout_seconds, "request_timeout_seconds", minimum=0.001)
    if not isinstance(config.agent_templates, dict):
        raise ValueError("agent_templates must map role IDs to instructions")
    if config.agent_templates:
        if len(config.agent_templates) < 2 or list(config.agent_templates)[-1] != "synthesizer":
            raise ValueError("agent_templates needs at least two roles, with synthesizer last")
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
               for k, v in config.agent_templates.items()):
            raise ValueError("Agent IDs and instructions must be nonempty strings")
    worker_count = len(config.agent_templates) - 1 if config.agent_templates else 2
    if isinstance(config.min_workers, bool) or not isinstance(config.min_workers, int) or not 1 <= config.min_workers <= worker_count:
        raise ValueError("min_workers must be between one and the number of candidate workers")
    return config
