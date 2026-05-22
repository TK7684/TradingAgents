"""Rate limiter for LLM API calls.

Implements a sliding-window rate limiter that all LLM clients share.
Prevents 429 errors by controlling request rate at the transport level.
"""
import threading
import time
import logging

log = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter using a sliding window.

    Tracks requests within a time window and blocks when the limit is reached.
    """

    def __init__(self, max_requests: int = 6, window_seconds: float = 60.0):
        """
        Args:
            max_requests: Maximum requests allowed in the window.
            window_seconds: Length of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list = []
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 120.0):
        """Block until a request can be made.

        Args:
            timeout: Maximum seconds to wait before raising.

        Raises:
            TimeoutError: If unable to acquire within timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                # Prune timestamps outside the window
                cutoff = now - self.window_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                # Calculate how long to wait for the oldest request to expire
                wait_time = self._timestamps[0] + self.window_seconds - now + 0.1

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Rate limiter timeout after {timeout}s: {self.max_requests} req/{self.window_seconds}s"
                )

            log.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(min(wait_time, 2.0))  # Check again in 2s max


# Global shared rate limiter instance for all LLM calls
_global_limiter = None
_global_lock = threading.Lock()


def get_rate_limiter(max_requests: int = 20, window_seconds: float = 60.0) -> RateLimiter:
    """Get or create the global rate limiter.

    Only creates a new instance on first call; subsequent calls return
    the existing instance regardless of arguments (prevents mid-session
    reconfiguration from creating multiple limiters).
    """
    global _global_limiter
    if _global_limiter is None:
        with _global_lock:
            if _global_limiter is None:
                _global_limiter = RateLimiter(max_requests, window_seconds)
                log.info(f"Rate limiter initialized: {max_requests} req / {window_seconds}s")
    return _global_limiter
