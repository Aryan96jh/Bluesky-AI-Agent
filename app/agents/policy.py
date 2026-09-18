class ActionPolicy:

    def __init__(
        self,
        min_relevance=70,
        min_opportunity=30,
        min_credibility=20,
        max_spam=60,
    ):
        self.min_relevance = min_relevance
        self.min_opportunity = min_opportunity
        self.min_credibility = min_credibility
        self.max_spam = max_spam

    def decide(self, analysis: dict) -> dict:

        if "error" in analysis:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Invalid AI response",
            }

        relevant = analysis.get(
            "relevant",
            False,
        )

        relevance = (
            analysis.get(
                "relevance_score",
                0,
            )
            or 0
        )

        content_value = (
            analysis.get(
                "content_value_score",
                0,
            )
            or 0
        )

        discussion = (
            analysis.get(
                "discussion_score",
                0,
            )
            or 0
        )

        credibility = (
            analysis.get(
                "credibility_score",
                0,
            )
            or 0
        )

        # IMPORTANT:
        # Do not use "or 100" here because spam_score=0
        # is a valid value and must remain 0.
        spam = analysis.get(
            "spam_score",
            100,
        )

        if spam is None:
            spam = 100

        risk = analysis.get(
            "risk_level",
            "high",
        )

        action = analysis.get(
            "action",
            "ignore",
        )

        reply = (
            analysis.get(
                "reply_idea",
                "",
            )
            or ""
        ).strip()

        opportunity = (
            content_value * 0.6
            + discussion * 0.4
        )

        if not relevant:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Post is not relevant",
            }

        if relevance < self.min_relevance:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Relevance score is too low",
            }

        if opportunity < self.min_opportunity:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Content/discussion opportunity is too low",
            }

        if credibility < self.min_credibility:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Credibility score is too low",
            }

        if spam > self.max_spam:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Post appears promotional or spammy",
            }

        if risk == "high":
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Post has high risk",
            }

        allowed_actions = {
            "reply",
            "quote",
            "post_idea",
            "ignore",
        }

        if action not in allowed_actions:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "Invalid action returned by AI",
            }

        if action == "ignore":
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "AI decided the post is not worth engaging with",
            }

        if action == "reply" and not reply:
            return {
                "allowed": False,
                "action": "ignore",
                "reason": "No usable reply generated",
            }

        return {
            "allowed": True,
            "action": action,
            "reason": (
                "Passed relevance, opportunity, credibility, "
                "spam, and risk checks"
            ),
        }
