import math
from datetime import datetime, timezone


class HypeDetector:
    def calculate_score(
        self,
        likes: int,
        replies: int,
        reposts: int,
        created_at: datetime,
    ) -> int:

        likes = max(likes, 0)
        replies = max(replies, 0)
        reposts = max(reposts, 0)

        engagement = (
            likes
            + (replies * 3)
            + (reposts * 4)
        )

        now = datetime.now(timezone.utc)

        if created_at.tzinfo is None:
            created_at = created_at.replace(
                tzinfo=timezone.utc
            )

        age_hours = max(
            (now - created_at).total_seconds() / 3600,
            0.1,
        )

        # Recency matters more for new conversations.
        if age_hours <= 1:
            recency_score = 100
        elif age_hours <= 3:
            recency_score = 90
        elif age_hours <= 6:
            recency_score = 80
        elif age_hours <= 12:
            recency_score = 65
        elif age_hours <= 24:
            recency_score = 50
        elif age_hours <= 48:
            recency_score = 30
        else:
            recency_score = 10

        # Engagement velocity.
        velocity = engagement / age_hours

        velocity_score = min(
            math.log10(velocity + 1) * 30,
            100,
        )

        engagement_score = min(
            math.log10(engagement + 1) * 25,
            100,
        )

        score = (
            recency_score * 0.45
            + velocity_score * 0.35
            + engagement_score * 0.20
        )

        return round(min(score, 100))
