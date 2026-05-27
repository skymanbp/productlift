"""Container healthcheck. FULLY BUILT. Stdlib only — a healthcheck must not fail
for its own reasons. Exits 0 if /health responds 200."""

from __future__ import annotations

import sys
import urllib.request

URL = "http://localhost:8000/health"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=3) as resp:  # noqa: S310 (local URL)
            return 0 if resp.status == 200 else 1
    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
