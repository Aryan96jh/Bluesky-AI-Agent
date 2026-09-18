import re


class AIAnalysisValidator:

    def __init__(
        self,
        min_discovery_score=60,
        min_technical_score=65,
        min_evidence_score=35,
        max_contradictory_relevance=20,
    ):
        self.min_discovery_score = min_discovery_score
        self.min_technical_score = min_technical_score
        self.min_evidence_score = min_evidence_score
        self.max_contradictory_relevance = max_contradictory_relevance

    def validate(
        self,
        analysis: dict,
        discovery: dict,
        post_text: str,
    ) -> dict:

        errors = []

        if not isinstance(analysis, dict):
            return {
                "valid": False,
                "reasons": [
                    "AI analysis is not a dictionary"
                ],
            }

        if not isinstance(discovery, dict):
            return {
                "valid": False,
                "reasons": [
                    "Discovery data is missing"
                ],
            }

        relevant = analysis.get("relevant")

        if not isinstance(relevant, bool):
            errors.append(
                "AI returned an invalid 'relevant' value"
            )

        action = analysis.get(
            "action",
            "ignore",
        )

        if action not in {
            "reply",
            "quote",
            "post_idea",
            "ignore",
        }:
            errors.append(
                f"AI returned invalid action: {action}"
            )

        numeric_fields = [
            "relevance_score",
            "content_value_score",
            "discussion_score",
            "credibility_score",
            "spam_score",
        ]

        for field in numeric_fields:

            value = analysis.get(
                field,
                0,
            )

            if value is None:
                value = 0

            if not isinstance(
                value,
                (int, float),
            ):
                errors.append(
                    f"{field} is not numeric"
                )
                continue

            if value < 0 or value > 100:
                errors.append(
                    f"{field} is outside 0-100 range"
                )

        discovery_score = float(
            discovery.get(
                "discovery_score",
                0,
            )
            or 0
        )

        technical_score = float(
            discovery.get(
                "technical_score",
                0,
            )
            or 0
        )

        evidence_score = float(
            discovery.get(
                "evidence_score",
                0,
            )
            or 0
        )

        strong_local_signal = (
            discovery_score
            >= self.min_discovery_score
            and technical_score
            >= self.min_technical_score
            and evidence_score
            >= self.min_evidence_score
        )

        ai_relevance = float(
            analysis.get(
                "relevance_score",
                0,
            )
            or 0
        )

        if strong_local_signal:

            if relevant is False:

                errors.append(
                    "AI marked a strongly relevant local candidate as irrelevant"
                )

            elif (
                ai_relevance
                <= self.max_contradictory_relevance
            ):

                errors.append(
                    "AI relevance is extremely low despite strong local discovery signals"
                )

        anchors = self._find_niche_anchors(
            post_text
        )

        if (
            len(anchors) >= 2
            and technical_score >= 65
            and ai_relevance <= 20
        ):

            errors.append(
                "Post contains multiple direct niche anchors but AI relevance is extremely low"
            )

        reply_idea = (
            analysis.get(
                "reply_idea",
                "",
            )
            or ""
        ).strip()

        if (
            action == "reply"
            and not reply_idea
        ):

            errors.append(
                "AI selected reply but generated no reply idea"
            )

        risk = analysis.get(
            "risk_level",
            "high",
        )

        if risk not in {
            "low",
            "medium",
            "high",
        }:

            errors.append(
                f"Invalid risk level: {risk}"
            )

        if errors:

            return {
                "valid": False,
                "reasons": errors,
                "discovery_score": discovery_score,
                "technical_score": technical_score,
                "evidence_score": evidence_score,
                "ai_relevance": ai_relevance,
                "anchors": anchors,
            }

        return {
            "valid": True,
            "reasons": [],
            "discovery_score": discovery_score,
            "technical_score": technical_score,
            "evidence_score": evidence_score,
            "ai_relevance": ai_relevance,
            "anchors": anchors,
        }

    @staticmethod
    def _find_niche_anchors(
        text: str,
    ) -> list[str]:

        if not text:
            return []

        text_lower = text.lower()

        anchor_groups = {

            "ai agents": [
                "ai agent",
                "ai agents",
                "agentic",
                "autonomous agent",
                "autonomous agents",
                "coding agent",
                "coding agents",
            ],

            "llm": [
                "llm",
                "large language model",
                "language model",
                "foundation model",
            ],

            "evaluation": [
                "benchmark",
                "evaluation",
                "evals",
                "evaluate",
                "testing",
            ],

            "reasoning": [
                "reasoning",
                "reasoning model",
                "chain of thought",
            ],

            "tool use": [
                "tool calling",
                "tool use",
                "function calling",
                "mcp",
                "model context protocol",
            ],

            "memory": [
                "agent memory",
                "memory system",
                "persistent memory",
                "long term memory",
            ],

            "automation": [
                "ai automation",
                "automation workflow",
                "agent workflow",
                "workflow",
            ],

            "inference": [
                "inference",
                "llm inference",
                "model inference",
            ],

            "open source ai": [
                "open source ai",
                "open-source ai",
                "open source model",
                "open-source model",
            ],

            "ai security": [
                "ai security",
                "agent security",
                "llm security",
                "prompt injection",
            ],
        }

        found = []

        for group, phrases in anchor_groups.items():

            for phrase in phrases:

                if re.search(
                    rf"\b{re.escape(phrase)}\b",
                    text_lower,
                ):

                    found.append(group)
                    break

        return found
