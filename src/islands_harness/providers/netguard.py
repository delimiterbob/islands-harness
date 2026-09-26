"""Runtime host allowlist transport and the owned retry policy.

Two guarantees the client-mode perimeter rests on (ARCHITECTURE.md D6, Section 13):

1. Every HTTP request the harness makes goes through ``AllowlistTransport``, which refuses
   any host not in the model's ``allowed_hosts`` before a socket is opened. This is enforced
   at runtime on the machine that runs the sweep, not only proved by a test on the commit.
2. The harness owns every retry. Both SDK-level retry counts are zero: the OpenAI-compatible
   adapter uses no SDK at all, and the Anthropic client is built with ``max_retries=0``.
   ``with_retry`` retries 429, 5xx, connection errors and timeouts with exponential backoff,
   at most six attempts, logging each as a ``transport_retry`` event, and raises
   ``TransportExhausted`` when they run out. Nothing else is ever retried.

HTTP library: ``httpx2`` (the maintained httpx fork the anthropic 1.x SDK is built on). The
SDK rejects transports and clients from the old ``httpx`` package, so one stack serves both
adapters: ``import httpx2 as httpx``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx2 as httpx


class HostNotAllowed(Exception):
    """A request targeted a host outside ``allowed_hosts``."""


class TransportError(Exception):
    """A transport failure classified by the adapter. ``retryable`` decides the policy."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class TransportExhausted(Exception):
    """The retry policy ran out. The run ends as ``aborted_transport``."""


def _host_key(url: httpx.URL) -> str:
    """``host`` or ``host:port`` as written in allowed_hosts. A bare host in the allowlist
    matches the scheme's default port only."""
    port = url.port
    default = 443 if url.scheme == "https" else 80
    return url.host if port is None or port == default else f"{url.host}:{port}"


class AllowlistTransport(httpx.AsyncBaseTransport):
    """An httpx2 async transport that refuses any host not in the allowlist and delegates
    everything else to a plain ``AsyncHTTPTransport`` with retries disabled."""

    def __init__(
        self, allowed_hosts: list[str], *, inner: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.allowed = frozenset(allowed_hosts)
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = _host_key(request.url)
        if key not in self.allowed and request.url.host not in self.allowed:
            raise HostNotAllowed(f"{key} is not in allowed_hosts {sorted(self.allowed)}")
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def build_http_client(allowed_hosts: list[str], timeout_s: float) -> httpx.AsyncClient:
    """An httpx2 client on the allowlist transport with one overall timeout."""
    return httpx.AsyncClient(
        transport=AllowlistTransport(allowed_hosts), timeout=httpx.Timeout(timeout_s)
    )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay(self, attempt: int) -> float:
        """Exponential backoff without jitter (jitter would be an unhashed random draw)."""
        return min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))


def classify_status(status: int) -> bool:
    """True when an HTTP status is retryable: 408, 409, 429 and every 5xx."""
    return status in (408, 409, 429) or 500 <= status <= 599


async def with_retry(
    fn: Callable[[], Awaitable[Any]],
    policy: RetryPolicy,
    log: Callable[[dict[str, Any]], None],
) -> Any:
    """Call ``fn`` until it returns, retrying only ``TransportError(retryable=True)``.

    Each retry logs ``{"event": "transport_retry", "attempt", "status", "delay_s", "error"}``
    through ``log``. A non-retryable TransportError propagates at once; exhaustion raises
    TransportExhausted. ``HostNotAllowed`` is never retried.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except TransportError as exc:
            if not exc.retryable or attempt >= policy.max_attempts:
                if exc.retryable:
                    raise TransportExhausted(f"gave up after {attempt} attempts: {exc}") from exc
                raise
            delay = policy.delay(attempt)
            log(
                {
                    "event": "transport_retry",
                    "attempt": attempt,
                    "status": exc.status,
                    "delay_s": delay,
                    "error": str(exc),
                }
            )
            await asyncio.sleep(delay)
