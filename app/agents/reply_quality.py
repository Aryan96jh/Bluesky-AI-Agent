import re


class ReplyQuality:

    def __init__(
        self,
        min_score: int = 70,
    ):
        self.min_score = min_score

    def evaluate(
        self,
        reply: str,
        post_text: str,
        analysis: dict | None = None,
    ) -> dict:

        analysis = analysis or {}

        reply = (
            reply
            or ""
        ).strip()

        if not reply:
            return {
                "approved": False,
                "score": 0,
                "reasons": [
                    "Reply is empty",
                ],
            }

        score = 50
        reasons = []

        word_count = self._word_count(reply)

        # --------------------------------------------------
        # LENGTH
        # --------------------------------------------------

        if word_count < 4:

            return {
                "approved": False,
                "score": 0,
                "reasons": [
                    "Reply is too short",
                ],
            }

        if word_count <= 18:

            score += 10

            reasons.append(
                "Good length"
            )

        else:

            score -= 25

            reasons.append(
                "Too long"
            )

        # --------------------------------------------------
        # GROUNDING
        # --------------------------------------------------

        grounding = self._grounding_score(
            reply,
            post_text,
        )

        if grounding >= 0.25:

            score += 10

            reasons.append(
                "Grounded in post"
            )

        else:

            score -= 15

            reasons.append(
                "Weak grounding"
            )

        # --------------------------------------------------
        # QUESTION QUALITY
        # --------------------------------------------------

        if "?" in reply:

            if self._specific_question(
                reply,
                post_text,
            ):

                score += 10

                reasons.append(
                    "Specific question"
                )

            else:

                score -= 20

                reasons.append(
                    "Vague question"
                )

        # --------------------------------------------------
        # GENERICITY
        # --------------------------------------------------

        if self._is_generic(reply):

            score -= 40

            reasons.append(
                "Generic or hype language"
            )

        # --------------------------------------------------
        # ENGAGEMENT BAIT
        # --------------------------------------------------

        if self._is_engagement_bait(reply):

            score -= 35

            reasons.append(
                "Engagement bait"
            )

        # --------------------------------------------------
        # HASHTAGS
        # --------------------------------------------------

        if "#" in reply:

            score -= 25

            reasons.append(
                "Contains hashtag"
            )

        # --------------------------------------------------
        # EMOJIS
        # --------------------------------------------------

        if self._contains_emoji(reply):

            score -= 10

            reasons.append(
                "Contains emoji"
            )

        # --------------------------------------------------
        # REPETITION
        # --------------------------------------------------

        repetition = self._repetition_score(
            reply,
            post_text,
        )

        if repetition >= 0.70:

            score -= 30

            reasons.append(
                "Mostly repeats the original post"
            )

        elif repetition >= 0.45:

            score -= 10

            reasons.append(
                "Some repetition of original post"
            )

        # --------------------------------------------------
        # TECHNICAL SIGNAL
        # --------------------------------------------------

        if self._has_technical_signal(
            reply,
            post_text,
        ):

            score += 10

            reasons.append(
                "Contains technical signal"
            )

        # --------------------------------------------------
        # SPECIFICITY
        # --------------------------------------------------

        if self._is_specific(reply):

            score += 5

            reasons.append(
                "Specific wording"
            )

        score = max(
            0,
            min(
                100,
                score,
            ),
        )

        approved = score >= self.min_score

        return {
            "approved": approved,
            "score": score,
            "reasons": reasons,
        }

    # ======================================================
    # WORD COUNT
    # ======================================================

    @staticmethod
    def _word_count(text: str) -> int:

        return len(
            re.findall(
                r"\b[\w’'-]+\b",
                text,
            )
        )

    # ======================================================
    # GROUNDING
    # ======================================================

    @staticmethod
    def _tokens(text: str) -> set[str]:

        words = re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9’'-]*\b",
            text.lower(),
        )

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "then",
            "than",
            "this",
            "that",
            "these",
            "those",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "for",
            "on",
            "in",
            "with",
            "from",
            "by",
            "at",
            "as",
            "it",
            "its",
            "they",
            "them",
            "their",
            "you",
            "your",
            "we",
            "our",
            "i",
            "me",
            "my",
            "how",
            "what",
            "why",
            "when",
            "where",
            "does",
            "do",
            "did",
            "can",
            "could",
            "would",
            "should",
        }

        return {
            word
            for word in words
            if word not in stopwords
            and len(word) >= 3
        }

    def _grounding_score(
        self,
        reply: str,
        post_text: str,
    ) -> float:

        reply_tokens = self._tokens(
            reply
        )

        post_tokens = self._tokens(
            post_text
        )

        if not reply_tokens:
            return 0.0

        overlap = (
            reply_tokens
            & post_tokens
        )

        return (
            len(overlap)
            / len(reply_tokens)
        )

    # ======================================================
    # REPETITION
    # ======================================================

    def _repetition_score(
        self,
        reply: str,
        post_text: str,
    ) -> float:

        reply_tokens = self._tokens(
            reply
        )

        post_tokens = self._tokens(
            post_text
        )

        if not reply_tokens:
            return 0.0

        overlap = (
            reply_tokens
            & post_tokens
        )

        return (
            len(overlap)
            / len(reply_tokens)
        )

    # ======================================================
    # GENERIC
    # ======================================================

    @staticmethod
    def _is_generic(reply: str) -> bool:

        text = reply.lower().strip()

        exact = {
            "great post",
            "great post!",
            "interesting",
            "interesting!",
            "very interesting",
            "very interesting!",
            "amazing",
            "amazing!",
            "awesome",
            "awesome!",
            "impressive",
            "impressive!",
            "love this",
            "love this!",
            "nice",
            "nice!",
            "wow",
            "wow!",
            "thoughts?",
            "any thoughts?",
            "what do you think?",
            "what's your take?",
            "whats your take?",
            "do you agree?",
            "agree?",
        }

        if text in exact:
            return True

        patterns = [
            r"^wow\b",
            r"^amazing\b",
            r"^awesome\b",
            r"^impressive\b",
            r"^great post\b",
            r"^interesting\b",
            r"^very interesting\b",
            r"^love this\b",
            r"^nice\b",
            r"\bgame changer\b",
            r"\bthis is huge\b",
            r"\bthis is insane\b",
            r"\bthis is exciting\b",
        ]

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )

    # ======================================================
    # ENGAGEMENT BAIT
    # ======================================================

    @staticmethod
    def _is_engagement_bait(
        reply: str,
    ) -> bool:

        text = reply.lower()

        patterns = [
            r"\bwhat do you think\b",
            r"\bwhat's your take\b",
            r"\bwhats your take\b",
            r"\bany thoughts\b",
            r"\bthoughts\?$",
            r"\bdo you agree\b",
            r"\bagree\?$",
            r"\banyone else\b",
            r"\blet me know\b",
            r"\bcurious what you think\b",
        ]

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )

    # ======================================================
    # QUESTION QUALITY
    # ======================================================

    def _specific_question(
        self,
        reply: str,
        post_text: str,
    ) -> bool:

        if "?" not in reply:
            return False

        question = reply.split(
            "?",
            1,
        )[0]

        question_tokens = self._tokens(
            question
        )

        post_tokens = self._tokens(
            post_text
        )

        if not question_tokens:
            return False

        overlap = (
            question_tokens
            & post_tokens
        )

        # A useful question should reference
        # something from the actual post.
        if len(overlap) < 1:
            return False

        vague_patterns = [
            r"what do you think",
            r"what's your take",
            r"whats your take",
            r"any thoughts",
            r"do you agree",
            r"agree",
        ]

        for pattern in vague_patterns:

            if re.search(
                pattern,
                question.lower(),
            ):
                return False

        return True

    # ======================================================
    # TECHNICAL SIGNAL
    # ======================================================

    def _has_technical_signal(
        self,
        reply: str,
        post_text: str,
    ) -> bool:

        technical_terms = {
            "api",
            "agent",
            "agents",
            "llm",
            "model",
            "models",
            "rag",
            "retrieval",
            "memory",
            "context",
            "tool",
            "tools",
            "mcp",
            "latency",
            "inference",
            "training",
            "benchmark",
            "benchmarks",
            "evaluation",
            "reasoning",
            "verification",
            "validation",
            "architecture",
            "database",
            "sqlite",
            "fts5",
            "pipeline",
            "coverage",
            "security",
            "reliability",
            "failure",
            "failures",
            "scaling",
            "voice",
            "coding",
        }

        tokens = self._tokens(
            reply
        )

        return bool(
            tokens
            & technical_terms
        )

    # ======================================================
    # SPECIFICITY
    # ======================================================

    def _is_specific(
        self,
        reply: str,
    ) -> bool:

        tokens = self._tokens(
            reply
        )

        if len(tokens) < 5:
            return False

        specific_terms = {
            "how",
            "why",
            "when",
            "backend",
            "frontend",
            "sqlite",
            "fts5",
            "rag",
            "retrieval",
            "latency",
            "verification",
            "benchmark",
            "pattern",
            "matching",
            "coverage",
            "drift",
            "memory",
            "tool",
            "execution",
            "architecture",
            "reasoning",
            "evaluation",
        }

        return bool(
            tokens
            & specific_terms
        )

    # ======================================================
    # EMOJI
    # ======================================================

    @staticmethod
    def _contains_emoji(
        text: str,
    ) -> bool:

        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]"
        )

        return bool(
            emoji_pattern.search(text)
        )
