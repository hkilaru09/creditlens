"""Self-imposed sliding-window rate limiter (independent of provider quotas)."""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period_seconds:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()

            self._calls.append(now)
