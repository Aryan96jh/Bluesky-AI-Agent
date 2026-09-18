import time
from datetime import datetime, timezone


class RateLimiter:
    """Runtime safety limits for bot action execution."""

    def __init__(
        self,
        memory,
        max_actions_per_day: int,
        delay_seconds: float,
        max_actions_per_run: int = 1,
        live_execution_enabled: bool = False,
    ):
        self.memory = memory
        self.max_actions_per_day = max(0, int(max_actions_per_day))
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.max_actions_per_run = max(0, int(max_actions_per_run))
        self.live_execution_enabled = bool(live_execution_enabled)

        self._last_action_at = None
        self._run_live_actions = 0

    def can_execute(self) -> tuple[bool, str]:
        if not self.live_execution_enabled:
            return (
                False,
                "LIVE execution is disabled by the safety gate",
            )

        if self.max_actions_per_day <= 0:
            return False, "Daily action limit is 0"

        if self.max_actions_per_run <= 0:
            return False, "Per-run action limit is 0"

        if self._run_live_actions >= self.max_actions_per_run:
            return (
                False,
                f"Per-run LIVE action limit reached "
                f"({self._run_live_actions}/{self.max_actions_per_run})",
            )

        today_count = self.memory.get_live_replies_today()

        if today_count >= self.max_actions_per_day:
            return (
                False,
                f"Daily LIVE action limit reached "
                f"({today_count}/{self.max_actions_per_day})",
            )

        return True, "Allowed"

    def wait_if_needed(self, dry_run: bool = False) -> None:
        if (
            dry_run
            or self.delay_seconds <= 0
            or self._last_action_at is None
        ):
            return

        elapsed = (
            datetime.now(timezone.utc) - self._last_action_at
        ).total_seconds()

        remaining = self.delay_seconds - elapsed

        if remaining > 0:
            print(
                f"[RATE LIMIT] Sleeping {remaining:.1f}s "
                "before next live action..."
            )
            time.sleep(remaining)

    def mark_executed(self) -> None:
        self._last_action_at = datetime.now(timezone.utc)
        self._run_live_actions += 1

    @property
    def run_live_actions(self) -> int:
        return self._run_live_actions
