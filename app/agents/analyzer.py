import re


class TweetAnalyzer:
    """
    AI analyzer for Bluesky posts.

    Compatible with the existing runner:
        analyzer.analyze(post_text=post.text)

    The model returns a compact JSON object to reduce
    token usage and avoid truncated responses.
    """

    def __init__(
        self,
        llm_client,
        niche,
        tone,
    ):
        self.llm = llm_client
        self.niche = niche
        self.tone = tone

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        post_text=None,
        post=None,
        activity_score=None,
    ):
        """
        Analyze a post.

        Supports both:
            analyze(post_text="...")
        and:
            analyze(post=post)

        This keeps compatibility with older code.
        """

        if post_text is None and post is not None:
            if isinstance(post, str):
                post_text = post
            else:
                post_text = getattr(
                    post,
                    "text",
                    "",
                )

        if post_text is None:
            post_text = ""

        post_text = str(
            post_text
        ).strip()

        if not post_text:
            return self._fallback_analysis(
                "Empty post."
            )

        prompt = self._build_prompt(
            text=post_text,
            activity_score=activity_score,
        )

        try:
            response = self.llm.generate_json(
                prompt,
                max_tokens=350,
                temperature=0.1,
            )

            analysis = self._normalize(
                response
            )

            analysis = (
                self._apply_deterministic_checks(
                    post_text,
                    analysis,
                )
            )

            if activity_score is not None:
                try:
                    analysis[
                        "activity_score"
                    ] = int(
                        activity_score
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    analysis[
                        "activity_score"
                    ] = 0
            else:
                analysis[
                    "activity_score"
                ] = 0

            # Final validation of the proposed
            # reply against the original post.
            if (
                analysis["action"] == "reply"
                and analysis["reply_idea"]
            ):
                if not self._validate_reply(
                    post_text,
                    analysis["reply_idea"],
                ):
                    analysis[
                        "action"
                    ] = "ignore"

                    analysis[
                        "reply_idea"
                    ] = ""

                    analysis[
                        "reason"
                    ] = (
                        "Proposed reply was "
                        "not sufficiently grounded "
                        "in the post."
                    )

            return analysis

        except Exception as error:
            print(
                f"[ANALYZER] Error: {error}"
            )

            return self._fallback_analysis(
                str(error)
            )

    # =========================================================
    # PROMPT
    # =========================================================

    def _build_prompt(
        self,
        text,
        activity_score=None,
    ):
        text = text[:3000]

        activity_line = ""

        if activity_score is not None:
            activity_line = (
                f"\nACTIVITY SCORE: "
                f"{activity_score}/100"
            )

        return f"""
You are the content quality classifier for a niche social-media bot.

NICHE:
{self.niche}

TONE:
{self.tone}
{activity_line}

POST:
{text}

Decide whether this post deserves a useful, non-spammy reply.

Focus on:
- AI agents
- autonomous AI systems
- LLMs
- AI automation
- agent frameworks
- MCP
- tool calling
- RAG
- reasoning
- inference
- coding agents
- benchmarks
- evaluation
- architecture
- reliability
- security
- AI research
- developer tooling
- meaningful technical discussion

IMPORTANT:
- Do not reward generic mentions of AI.
- Prefer concrete technical substance.
- Ignore vague engagement bait.
- Ignore obvious promotion.
- Ignore low-information posts.
- Do not invent facts.
- Do not make claims that are not supported by the post.
- High-risk or unverified serious claims should normally be ignored.
- Only choose reply when there is a clear useful angle.
- The reply idea must be grounded in the actual post.

RETURN ONLY ONE JSON OBJECT.

Use EXACTLY these keys:

{{
  "r": true,
  "rel": 0,
  "val": 0,
  "disc": 0,
  "cred": 0,
  "spam": 0,
  "risk": "low",
  "act": "reply",
  "idea": "short grounded reply"
}}

Definitions:

r = relevant
rel = niche relevance
val = content value
disc = discussion potential
cred = apparent credibility
spam = spam/promotional level
risk = low, medium, or high
act = reply or ignore
idea = reply under 240 characters

Rules:
- Numbers must be integers from 0 to 100.
- Keep idea under 240 characters.
- Keep idea specific to the post.
- No hashtags.
- No emojis.
- No generic praise.
- No "great post".
- No "interesting".
- No "thanks for sharing".
- No markdown.
- No explanation outside JSON.
- If act is ignore, idea must be empty.
- If risk is high, act must be ignore.
""".strip()

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize(
        self,
        data,
    ):
        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Analyzer response must be a JSON object."
            )

        # Support compact format.
        relevant = data.get(
            "r",
            data.get(
                "relevant",
                False,
            ),
        )

        relevance = data.get(
            "rel",
            data.get(
                "relevance_score",
                0,
            ),
        )

        value = data.get(
            "val",
            data.get(
                "content_value_score",
                0,
            ),
        )

        discussion = data.get(
            "disc",
            data.get(
                "discussion_score",
                0,
            ),
        )

        credibility = data.get(
            "cred",
            data.get(
                "credibility_score",
                0,
            ),
        )

        spam = data.get(
            "spam",
            data.get(
                "spam_score",
                0,
            ),
        )

        risk = data.get(
            "risk",
            data.get(
                "risk_level",
                "high",
            ),
        )

        action = data.get(
            "act",
            data.get(
                "action",
                "ignore",
            ),
        )

        idea = data.get(
            "idea",
            data.get(
                "reply_idea",
                "",
            ),
        )

        reason = data.get(
            "reason",
            "AI classification",
        )

        analysis = {
            "relevant": bool(
                relevant
            ),
            "relevance_score": self._score(
                relevance
            ),
            "content_value_score": self._score(
                value
            ),
            "discussion_score": self._score(
                discussion
            ),
            "credibility_score": self._score(
                credibility
            ),
            "spam_score": self._score(
                spam
            ),
            "risk_level": self._risk(
                risk
            ),
            "action": (
                "reply"
                if str(action).lower()
                == "reply"
                else "ignore"
            ),
            "reason": str(
                reason
            )[:500],
            "reply_idea": self._clean_reply(
                idea
            ),
        }

        analysis[
            "opportunity_score"
        ] = self._calculate_opportunity(
            analysis
        )

        # Safety rules.
        if not analysis[
            "relevant"
        ]:
            analysis[
                "action"
            ] = "ignore"

            analysis[
                "reply_idea"
            ] = ""

        if analysis[
            "risk_level"
        ] == "high":
            analysis[
                "action"
            ] = "ignore"

            analysis[
                "reply_idea"
            ] = ""

        return analysis

    # =========================================================
    # SCORE HELPERS
    # =========================================================

    @staticmethod
    def _score(
        value,
    ):
        try:
            value = int(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0

        return max(
            0,
            min(
                value,
                100,
            ),
        )

    @staticmethod
    def _risk(
        value,
    ):
        value = str(
            value
        ).lower().strip()

        if value not in {
            "low",
            "medium",
            "high",
        }:
            return "high"

        return value

    @staticmethod
    def _clean_reply(
        value,
    ):
        if value is None:
            return ""

        reply = str(
            value
        ).strip()

        reply = reply.replace(
            "\n",
            " ",
        )

        reply = re.sub(
            r"\s+",
            " ",
            reply,
        )

        return reply[:240]

    # =========================================================
    # OPPORTUNITY
    # =========================================================

    @staticmethod
    def _calculate_opportunity(
        analysis,
    ):
        relevance = analysis[
            "relevance_score"
        ]

        value = analysis[
            "content_value_score"
        ]

        discussion = analysis[
            "discussion_score"
        ]

        credibility = analysis[
            "credibility_score"
        ]

        spam = analysis[
            "spam_score"
        ]

        opportunity = (
            relevance * 0.30
            + value * 0.25
            + discussion * 0.25
            + credibility * 0.20
            - spam * 0.20
        )

        return max(
            0,
            min(
                round(opportunity),
                100,
            ),
        )

    # =========================================================
    # DETERMINISTIC CHECKS
    # =========================================================

    def _apply_deterministic_checks(
        self,
        text,
        analysis,
    ):
        lower = text.lower()

        strong_niche_signals = [
            "ai agent",
            "ai agents",
            "autonomous agent",
            "autonomous agents",
            "llm",
            "large language model",
            "mcp",
            "model context protocol",
            "agentic",
            "agentic ai",
            "tool calling",
            "function calling",
            "rag",
            "retrieval augmented",
            "context window",
            "coding agent",
            "coding agents",
            "agent framework",
            "agent frameworks",
            "llm inference",
            "llm reasoning",
            "ai automation",
            "ai workflow",
            "agent security",
            "agent safety",
            "agent evaluation",
            "ai benchmark",
            "llm benchmark",
        ]

        signal_hits = sum(
            1
            for term in strong_niche_signals
            if term in lower
        )

        if signal_hits >= 1:
            analysis[
                "relevance_score"
            ] = max(
                analysis[
                    "relevance_score"
                ],
                60,
            )

        if signal_hits >= 2:
            analysis[
                "relevance_score"
            ] = max(
                analysis[
                    "relevance_score"
                ],
                70,
            )

        # Engagement bait.
        bait_terms = [
            "what do you think",
            "thoughts?",
            "agree?",
            "agree or disagree",
            "drop your",
            "comment below",
            "who agrees",
        ]

        bait_hits = sum(
            1
            for term in bait_terms
            if term in lower
        )

        if bait_hits:
            analysis[
                "spam_score"
            ] = min(
                100,
                analysis[
                    "spam_score"
                ]
                + 10 * bait_hits,
            )

        # Promotional language.
        promotional_terms = [
            "buy now",
            "sign up",
            "limited time",
            "free trial",
            "use my link",
            "book a demo",
            "our product",
            "our platform",
            "our service",
            "skyrocket your",
        ]

        promo_hits = sum(
            1
            for term in promotional_terms
            if term in lower
        )

        if promo_hits:
            analysis[
                "spam_score"
            ] = min(
                100,
                analysis[
                    "spam_score"
                ]
                + 15 * promo_hits,
            )

        # Very short posts have limited
        # substantive discussion potential.
        if len(
            text.strip()
        ) < 50:
            analysis[
                "content_value_score"
            ] = min(
                analysis[
                    "content_value_score"
                ],
                25,
            )

        # Recalculate.
        analysis[
            "opportunity_score"
        ] = self._calculate_opportunity(
            analysis
        )

        # Final deterministic gates.
        if analysis[
            "spam_score"
        ] >= 60:
            analysis[
                "action"
            ] = "ignore"

            analysis[
                "reply_idea"
            ] = ""

        if analysis[
            "relevance_score"
        ] < 50:
            analysis[
                "action"
            ] = "ignore"

            analysis[
                "reply_idea"
            ] = ""

        return analysis

    # =========================================================
    # REPLY VALIDATION
    # =========================================================

    @staticmethod
    def _important_tokens(
        text,
    ):
        if not text:
            return set()

        words = re.findall(
            r"[a-zA-Z0-9]{4,}",
            text.lower(),
        )

        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "they",
            "them",
            "have",
            "will",
            "would",
            "could",
            "should",
            "about",
            "there",
            "their",
            "what",
            "when",
            "where",
            "which",
            "your",
            "just",
            "into",
            "than",
            "then",
            "also",
            "very",
            "more",
            "some",
            "does",
            "doesn",
            "been",
            "being",
        }

        return {
            word
            for word in words
            if word not in stopwords
        }

    @classmethod
    def _validate_reply(
        cls,
        post_text,
        reply,
    ):
        if not reply:
            return False

        if len(
            reply
        ) > 240:
            return False

        post_tokens = cls._important_tokens(
            post_text
        )

        reply_tokens = cls._important_tokens(
            reply
        )

        if not post_tokens:
            return False

        overlap = (
            post_tokens
            & reply_tokens
        )

        # Require at least one meaningful
        # token from the actual post.
        if len(overlap) >= 1:
            return True

        # Questions can sometimes be grounded
        # without repeating a specific term,
        # but we keep this conservative.
        return False

    # =========================================================
    # FALLBACK
    # =========================================================

    @staticmethod
    def _fallback_analysis(
        reason,
    ):
        return {
            "relevant": False,
            "relevance_score": 0,
            "content_value_score": 0,
            "discussion_score": 0,
            "credibility_score": 0,
            "spam_score": 100,
            "opportunity_score": 0,
            "risk_level": "high",
            "action": "ignore",
            "reason": str(
                reason
            )[:500],
            "reply_idea": "",
            "activity_score": 0,
        }
