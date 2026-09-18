from dataclasses import dataclass
from typing import Any


@dataclass
class QueuedAction:
    action: str
    post: Any
    text: str
    analysis: dict


class ActionQueue:
    """Small in-memory queue for approved bot actions."""

    def __init__(self):
        self._items: list[QueuedAction] = []

    def enqueue(self, action: str, post: Any, text: str, analysis: dict) -> None:
        self._items.append(
            QueuedAction(
                action=action,
                post=post,
                text=text,
                analysis=analysis,
            )
        )

    def __len__(self) -> int:
        return len(self._items)

    def pop_next(self):
        if not self._items:
            return None
        return self._items.pop(0)

    def clear(self) -> None:
        self._items.clear()

    def items(self) -> list[QueuedAction]:
        return list(self._items)
