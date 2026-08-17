"""Baseline HTTP security headers.

Audit finding: production answered with none of these. TLS itself is
sound (TLS 1.3, valid chain, terminated at Railway's edge) — what was
missing was everything that tells a browser how to treat the response.

This is a JSON API, not a site, so the policy is maximally restrictive:
nothing should ever be loaded, framed, or sniffed out of a response
here. The one exception is the Swagger/ReDoc UI, which only exists in
local and test (see create_app) and legitimately needs to load its own
scripts and styles — CSP is skipped for those two paths rather than
loosened for everyone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

__all__ = ["SECURITY_HEADERS", "SecurityHeadersMiddleware"]


# `default-src 'none'` is the right default for an API: a JSON body has
# no legitimate reason to pull a script, style, image or font. Combined
# with frame-ancestors 'none' it also makes clickjacking moot for any
# HTML we might accidentally return.
_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

SECURITY_HEADERS: dict[str, str] = {
    # 2 years, subdomains included. Sent only over HTTPS (see below) —
    # a browser must not pin HSTS from a plaintext localhost response.
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    # Stop content-type sniffing: a JSON body must never be executed as
    # script because a browser guessed.
    "X-Content-Type-Options": "nosniff",
    # Legacy sibling of frame-ancestors, for clients predating CSP2.
    "X-Frame-Options": "DENY",
    # Don't leak API paths (which embed patient/camera UUIDs) to any
    # third-party origin a response might link to.
    "Referrer-Policy": "no-referrer",
    # This API needs none of these device capabilities.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": _CSP,
}

# Swagger UI and ReDoc load their own bundles; a `default-src 'none'`
# page renders blank. These routes exist only in local/test.
_CSP_EXEMPT_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach `SECURITY_HEADERS` to every response.

    Existing headers win: a handler that deliberately sets its own CSP
    (none do today) should not be silently overridden by middleware.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        path = request.url.path
        for name, value in SECURITY_HEADERS.items():
            if name in response.headers:
                continue
            if name == "Content-Security-Policy" and path in _CSP_EXEMPT_PATHS:
                continue
            if name == "Strict-Transport-Security" and not _is_https(request):
                continue
            response.headers[name] = value
        return response


def _is_https(request: Request) -> bool:
    """True when the ORIGINAL client request used TLS.

    Railway terminates TLS at the edge and forwards plaintext, so
    `request.url.scheme` is "http" in production. X-Forwarded-Proto
    carries the truth; trusting it is safe here because the only
    consequence of a spoofed value is sending HSTS to a client that
    already claimed to be on HTTPS.
    """
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"
