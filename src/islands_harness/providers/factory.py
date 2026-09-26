"""Build a provider, and the per-run Sampling, from a ModelConfig.

One function per concern, so the only place a config turns into a live connection is here.
"""

from __future__ import annotations

from typing import Any

from islands_harness.config import AgentConfig, ModelConfig
from islands_harness.providers.base import ChatProvider, Sampling


def sampling_for(model: ModelConfig, seed: int | None) -> Sampling:
    """The Sampling sent for one run: ``seed: per_run`` takes the run's derived seed, an
    integer in the config is used as is, and null sends no seed."""
    s = model.sampling
    if s.seed == "per_run":
        run_seed = seed
    elif isinstance(s.seed, int):
        run_seed = s.seed
    else:
        run_seed = None
    return Sampling(
        temperature=s.temperature,
        top_p=s.top_p,
        max_tokens=s.max_tokens,
        seed=run_seed,
        thinking=s.thinking,
        effort=s.effort,
    )


def build_provider(model: ModelConfig, agent: AgentConfig, *, log: Any = None) -> ChatProvider:
    """A live provider for ``model``. Capability checks run at construction, so a config that
    asks a provider for a field it refuses fails here, before any request."""
    if model.provider == "openai_compat":
        from islands_harness.providers.openai_compat import OpenAICompatProvider

        if not model.base_url:
            raise ValueError(f"model {model.id!r}: openai_compat needs base_url")
        return OpenAICompatProvider(
            model.base_url,
            model.model,
            api_key_env=model.api_key_env,
            sampling=sampling_for(model, None),
            extra_body=model.extra_body,
            allowed_hosts=model.allowed_hosts,
            parallel_tool_calls=agent.parallel_tool_calls == "allow",
            log=log,
        )
    if model.provider == "anthropic":
        from islands_harness.providers.anthropic_native import AnthropicProvider

        # Raises ValueError when the key's environment variable is unset: without an explicit
        # key the SDK would fetch credentials through its own client, outside the allowlist.
        return AnthropicProvider(
            model.model,
            sampling=sampling_for(model, None),
            prompt_caching=model.prompt_caching,
            allowed_hosts=model.allowed_hosts,
            transport=model.transport or "anthropic",
            api_key_env=model.api_key_env,
            transport_options=model.transport_options,
            parallel_tool_calls=agent.parallel_tool_calls == "allow",
            log=log,
        )
    raise ValueError(f"model {model.id!r}: unknown provider {model.provider!r}")
