"""Protection for a public deployment: per-client hourly quotas and a cap on concurrent heavy
jobs. In-process (one API instance), which is all a single small container needs."""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import HTTPException, Request


def client_id(request: Request) -> str:
    """Real client address behind the host's proxy (Fly/Cloud Run/Render set these headers)."""
    for header in ("fly-client-ip", "x-forwarded-for", "x-real-ip"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class Limiter:
    def __init__(self, per_hour: dict[str, int], concurrent: dict[str, int]):
        self.per_hour = per_hour
        self.slots = {kind: threading.BoundedSemaphore(n) for kind, n in concurrent.items()}
        self.calls: dict[tuple[str, str], deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, request: Request, kind: str) -> None:
        limit = self.per_hour.get(kind)
        if not limit:
            return
        key, now = (client_id(request), kind), time.monotonic()
        with self.lock:
            calls = self.calls[key]
            while calls and now - calls[0] > 3600:
                calls.popleft()
            if len(calls) >= limit:
                retry = int(3600 - (now - calls[0])) + 1
                raise HTTPException(429, f"Hourly limit of {limit} {kind} requests reached; "
                                         f"try again in {retry // 60 + 1} min.",
                                    headers={"Retry-After": str(retry)})
            calls.append(now)

    @contextmanager
    def slot(self, kind: str):
        sem = self.slots.get(kind)
        if sem is None:
            yield
            return
        if not sem.acquire(blocking=False):
            raise HTTPException(503, f"The server is busy with another {kind} run; try again in a "
                                     "minute.", headers={"Retry-After": "30"})
        try:
            yield
        finally:
            sem.release()
