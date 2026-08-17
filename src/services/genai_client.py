"""Shared google-genai client.

Every Gemini call site used to construct its own ``genai.Client``
inline. Each construction eagerly builds a sync ``httpx.Client`` AND an
``httpx.AsyncClient`` (connection pool, TLS session, SSL buffers), none
of it ever closed — at one client per detection tick plus one per
redaction chunk that was 1,000+ abandoned pools per hour under load,
kept alive until a gen-2 GC found their reference cycles. A connection
pool is exactly the thing that should live for the process, not the
call.

One cached client per API key. The cache is bounded (an installation
has one or two keys, ever) and the client is safe to share: all callers
run on the single application event loop, and google-genai's async
surface is stateless per request.
"""

from __future__ import annotations

from functools import lru_cache

from google import genai


@lru_cache(maxsize=4)
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)
