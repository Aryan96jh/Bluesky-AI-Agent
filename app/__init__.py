"""NicheBot app package.

Sanitizes proxy env vars on import: some sandboxed environments export
no_proxy entries (e.g. bracketed IPv6 like "[::1]") that this httpx version
fails to parse ("Invalid port: ':1]'") at client construction time.
Unbracketed forms are kept.
"""

import os


def _sanitize_proxy_env() -> None:
    for key in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(key)
        if not val:
            continue
        cleaned = ",".join(
            part
            for part in val.split(",")
            if not (part.startswith("[") and part.endswith("]"))
        )
        os.environ[key] = cleaned


_sanitize_proxy_env()
