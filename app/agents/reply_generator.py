import hashlib
import re


class ReplyGenerator:
    """
    Generates short, natural replies grounded in the source post.

    v18 improvements:
    - Multiple reply styles
    - Short-post handling
    - Stronger grounding for named entities / compounds
    - Technical-family grounding
    - Strict question grounding
    - Reduced unsupported inference
    - Stronger claim validation
    - Softer handling of contested claims
    - Reduced repetitive reply patterns
    - Automatic retry when validation fails
    - Better source-term handling for hyphenated technical vocabulary
    - Stronger protection against unsupported paraphrase/causal claims
    - More specific grounded fallbacks for research posts
    """

    MAX_RETRIES = 2
    MAX_WORDS = 28
    TARGET_MAX_WORDS = 24
    MIN_WORDS = 5

    REPLY_STYLES = [
        "observation",
        "technical_question",
        "follow_up",
        "concise_insight",
        "counterpoint",
    ]

    # ------------------------------------------------------------------
    # TECHNICAL TERMS
    # ------------------------------------------------------------------

    TECHNICAL_TERMS = {
        "llm",
        "llms",
        "ai",
        "agent",
        "agents",
        "mcp",
        "rag",
        "embedding",
        "embeddings",
        "transformer",
        "transformers",
        "token",
        "tokens",
        "inference",
        "fine-tuning",
        "finetuning",
        "reinforcement",
        "rl",
        "rlhf",
        "benchmark",
        "benchmarks",
        "evaluation",
        "evals",
        "orchestration",
        "multi-agent",
        "multiagent",
        "context",
        "context-window",
        "memory",
        "vector",
        "vectors",
        "retrieval",
        "retrieval-augmented",
        "retrieval-augmented-generation",
        "reasoning",
        "tool-calling",
        "tool-use",
        "zero-shot",
        "few-shot",
        "protocol",
        "protocols",
        "sandbox",
        "sandboxed",
        "unsandboxed",
        "deployment",
        "deployments",
        "inference-time",
        "latency",
        "throughput",
        "mode-collapse",
        "mode",
        "gradient",
        "gradients",
        "policy-gradient",
        "policy-gradients",
        "neural",
        "neural-network",
        "neural-networks",
        "architecture",
        "architectures",
        "database",
        "databases",
        "persistence",
        "persistent",
        "cache",
        "caching",
        "api",
        "apis",
        "security",
        "authorization",
        "authentication",
        "auth",
        "alignment",
        "interpretability",
        "oversight",
        "hallucination",
        "hallucinations",
        "training",
        "fine-tuning",
        "reasoning-time",
        "planning",
        "planner",
        "planners",
        "workflow",
        "workflows",
        "rpc",
        "microservice",
        "microservices",
        "tokenizer",
        "tokenizers",
        "softmax",
        "confidence-threshold",
        "confidence-thresholds",
        "parallelism",
        "distributed",
        "distribution",
        "docs",
        "documentation",
        "tool-output",
        "tool-outputs",
        "feedback-loop",
        "feedback-loops",
        "editorial",
        "editorial-voice",
        "mimic",
        "mimics",
        "recent",
        "human",

        # Additional concrete technical concepts that must be explicitly
        # present in the source before the model may introduce them.
        "iot",
        "iot-device",
        "iot-devices",
        "zero-trust",
        "anomaly",
        "anomalies",
        "anomaly-detection",
        "voice-command",
        "voice-commands",
        "controller",
        "controllers",
        "rbac",
        "owasp",
        "vulnerability",
        "vulnerabilities",
        "injection",
        "device-control",
        "smart-home",
    }

    # Technical concepts that can reasonably refer to one another.
    # This allows "persistent context" and "memory" to be understood
    # as related without allowing completely unrelated concepts.
    TERM_FAMILIES = [
        {
            "llm",
            "llms",
            "model",
            "models",
            "ai",
        },
        {
            "agent",
            "agents",
            "multi-agent",
            "multiagent",
            "orchestration",
            "planning",
            "planner",
            "planners",
        },
        {
            "benchmark",
            "benchmarks",
            "evaluation",
            "evals",
        },
        {
            "security",
            "authorization",
            "authentication",
            "auth",
        },
        {
            "retrieval",
            "rag",
            "retrieval-augmented",
            "retrieval-augmented-generation",
            "embedding",
            "embeddings",
            "vector",
            "vectors",
        },
        {
            "inference",
            "inference-time",
            "latency",
            "throughput",
            "reasoning-time",
        },
        {
            "context",
            "context-window",
            "memory",
            "persistence",
            "persistent",
            "cache",
            "caching",
        },
        {
            "tool-calling",
            "tool-use",
            "mcp",
            "protocol",
            "protocols",
            "rpc",
        },
        {
            "architecture",
            "architectures",
            "workflow",
            "workflows",
            "orchestration",
        },
        {
            "database",
            "databases",
            "persistence",
            "persistent",
            "cache",
            "caching",
        },
    ]

    # ------------------------------------------------------------------
    # CLAIM WORDS
    # ------------------------------------------------------------------

    CLAIM_WORDS = {
        "is",
        "are",
        "was",
        "were",
        "will",
        "can",
        "does",
        "do",
        "prevents",
        "prevent",
        "reduces",
        "reduce",
        "improves",
        "improve",
        "causes",
        "cause",
        "means",
        "shows",
        "show",
        "proves",
        "prove",
        "ensures",
        "ensure",
        "guarantees",
        "guarantee",
        "makes",
        "make",
        "leads",
        "lead",
        "results",
        "result",
        "works",
        "work",
        "solves",
        "solve",
        "fixes",
        "fix",
    }

    STRONG_CLAIM_PATTERNS = [
        r"\bproves?\b",
        r"\bguarantee(?:s|d)?\b",
        r"\bdefinitively\b",
        r"\bclearly\b",
        r"\bdefinitely\b",
        r"\bcauses?\b",
        r"\bprevents?\b",
        r"\bensures?\b",
        r"\bsolves?\b",
        r"\bfixes?\b",
    ]

    CAUSAL_PATTERNS = [
        r"\bbecause\b",
        r"\bdue to\b",
        r"\bby\s+\w+ing\b",
        r"\bleading to\b",
        r"\bwhich means\b",
        r"\bthis causes\b",
        r"\bthis prevents\b",
        r"\bthis ensures\b",
        r"\bresults in\b",
    ]

    # Posts involving elections, parties, politicians, legislation, or
    # public-policy disputes need extra-neutral handling. The bot may
    # discuss what the source says, but should not endorse, oppose, rank,
    # or campaign for a political position.
    POLITICAL_MARKERS = {
        "election", "elections", "vote", "voting", "ballot", "candidate",
        "candidates", "campaign", "campaigns", "party", "parties",
        "politician", "politicians", "president", "prime minister",
        "parliament", "congress", "senate", "government", "governor",
        "legislation", "bill", "bills", "law", "laws", "policy", "policies",
        "regulation", "regulations", "democrat", "republican", "labour",
        "labor", "conservative", "liberal", "green", "socialist",
        "energy security", "public interests", "tax", "taxes",
        "immigration", "geopolitics", "sanctions", "war", "ukraine",
        "russia", "israel", "gaza", "palestine",
    }

    POLITICAL_ENDORSEMENT_PATTERNS = [
        r"\b(i|we)\s+(support|oppose|endorse|reject)\b",
        r"\b(vote|voting)\s+(for|against)\b",
        r"\bshould\s+vote\b",
        r"\bmust\s+vote\b",
        r"\bthe\s+right\s+choice\b",
        r"\bthe\s+best\s+(candidate|party|policy|option)\b",
        r"\bthe\s+worst\s+(candidate|party|policy|option)\b",
        r"\b(they|he|she)\s+should\s+win\b",
        r"\bwill\s+win\b",
        r"\bwould\s+win\b",
        r"\bmore\s+likely\s+to\s+win\b",
        r"\bvote\s+yes\b",
        r"\bvote\s+no\b",
    ]

    POLITICAL_NEUTRAL_PATTERNS = [
        r"\bthe\s+post\s+raises\b",
        r"\bthe\s+claim\s+is\b",
        r"\bthe\s+tradeoff\b",
        r"\bthe\s+distinction\b",
        r"\bthe\s+question\b",
        r"\baccording\s+to\b",
        r"\bthe\s+framing\b",
        r"\bthis\s+raises\s+an\s+interesting\s+question\b",
    ]

    # Phrases that commonly make generated replies sound like generic
    # AI commentary. These target openings/templates, not normal uses of
    # words such as "interesting" later in a sentence.
    ROBOTIC_OPENING_PATTERNS = [
        r"^this is (?:really |very |quite )?(?:interesting|fascinating|notable)\\b",
        r"^very interesting\\b",
        r"^really interesting\\b",
        r"^interesting[—,-]\\s*",
        r"^the (?:key|main|important) (?:point|thing|part) is\\b",
        r"^the (?:post|article|thread) (?:highlights|underscores|raises)\\b",
        r"^this (?:highlights|underscores|raises)\\b",
        r"^that framing\\b",
        r"^the distinction here\\b",
        r"^the tradeoff here\\b",
        r"^the .+ angle is\\b",
        r"^the .+ part is\\b",
    ]

    GENERIC_NATURALNESS_PATTERNS = [
        r"\\badds useful specificity\\b",
        r"\\bworth exploring in more detail\\b",
        r"\\bworth watching\\b",
        r"\\bworth considering\\b",
    ]

    MARKETING_PATTERNS = [
        r"\b(?:sign up|signup|register|download|book a demo|request a demo)\b",
        r"\blearn more\b",
        r"\b(?:buy now|use my code|limited time|free trial|discount)\b",
        r"\b(?:check out|try)\s+(?:our|my|this|the)\s+(?:product|platform|tool|service|app|solution)\b",
        r"\b(?:best|great|excellent|amazing|fantastic|powerful)\s+(?:product|platform|tool|service|solution)\b",
    ]

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, llm_client, niche=None, **kwargs):
        self.llm = llm_client
        self.niche = niche or ""

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate(self, post_text, analysis=None):
        """Compatibility wrapper used by runner.py."""
        return self.generate_reply(post_text, analysis)

    def generate_reply(
        self,
        post_text: str,
        analysis: dict | None = None,
    ) -> str:
        """
        Generate a validated reply.

        Returns:
            Valid reply string.

        Raises:
            ValueError if no valid reply can be generated.
        """

        post_text = (post_text or "").strip()

        if not post_text:
            raise ValueError(
                "Cannot generate reply for empty post"
            )

        analysis = analysis or {}

        short_post = self.is_short_post(post_text)

        for attempt in range(self.MAX_RETRIES + 1):

            style = self.choose_style(
                post_text=post_text,
                analysis=analysis,
                attempt=attempt,
            )

            prompt = self.build_prompt(
                post_text=post_text,
                analysis=analysis,
                style=style,
                short_post=short_post,
                retry=attempt > 0,
            )

            try:
                reply = self.llm.generate(prompt)

            except Exception as e:

                if attempt >= self.MAX_RETRIES:
                    print(
                        "[REPLY GENERATOR] LLM generation failed after "
                        f"{self.MAX_RETRIES + 1} attempts: {e}"
                    )
                    return ""

                continue

            reply = self.clean_reply(reply)

            if reply.strip().upper() == "SKIP":
                print(
                    f"[REPLY GENERATOR] Style={style} "
                    f"Attempt={attempt + 1} Rejected: LLM requested SKIP"
                )
                continue

            valid, reason = self.validate_reply(
                reply=reply,
                post_text=post_text,
                analysis=analysis,
                short_post=short_post,
            )

            if valid:

                print(
                    f"[REPLY GENERATOR] "
                    f"Style={style} "
                    f"Attempt={attempt + 1} "
                    f"Valid=True"
                )

                return reply

            print(
                f"[REPLY GENERATOR] "
                f"Style={style} "
                f"Attempt={attempt + 1} "
                f"Rejected: {reason}"
            )

        fallback = self.build_fallback_reply(post_text, analysis)
        if fallback:
            valid, reason = self.validate_reply(
                reply=fallback,
                post_text=post_text,
                analysis=analysis,
                short_post=short_post,
            )
            if valid:
                print(
                    "[REPLY GENERATOR] "
                    "Fallback=grounded_observation "
                    "Valid=True"
                )
                return fallback
            print(
                "[REPLY GENERATOR] Fallback rejected: "
                f"{reason}"
            )

        print(
            "[REPLY GENERATOR] No grounded reply could be generated after "
            f"{self.MAX_RETRIES + 1} attempts; skipping candidate."
        )
        return ""

    # ------------------------------------------------------------------
    # DETERMINISTIC FALLBACK
    # ------------------------------------------------------------------

    @staticmethod
    def build_fallback_reply(post_text: str, analysis: dict | None = None) -> str:
        """Do not manufacture a reply when the LLM cannot stay grounded."""
        return ""

    def generate(
        self,
        post_text: str,
        analysis: dict | None = None,
    ) -> str:
        return self.generate_reply(post_text, analysis)

    def generate_reply_idea(
        self,
        post_text: str,
        analysis: dict | None = None,
    ) -> str:
        return self.generate_reply(post_text, analysis)

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------

    def _source_grounding_guide(self, post_text: str) -> str:
        """Build a compact whitelist of source-supported technical language."""
        text = post_text or ""
        terms = sorted(self.extract_technical_terms(text.lower()))

        # Preserve useful hashtags such as #InferenceOptimization by
        # explicitly exposing the recognizable technical component.
        hashtag_terms = []
        for tag in re.findall(r"#([A-Za-z][A-Za-z0-9_-]*)", text):
            tag_lower = tag.lower()
            for term in self.TECHNICAL_TERMS:
                if len(term) >= 4 and term.replace("-", "") in tag_lower.replace("-", ""):
                    hashtag_terms.append(term)

        all_terms = sorted(set(terms + hashtag_terms))
        if not all_terms:
            return "No technical vocabulary whitelist available; use only exact source wording."

        return (
            "SOURCE-SUPPORTED TECHNICAL TERMS (whitelist): "
            + ", ".join(all_terms)
            + "\nIf a technical term is not on this list and is not ordinary language, do not introduce it."
            + "\nTechnical grounding is exact: do not substitute a related concept from the same technical family."
            + "\nFor example, source ``retrieval`` does not authorize ``RAG``; source ``tools`` does not authorize ``tool-calling``; source ``agents`` does not authorize ``sandbox``."
        )

    def build_prompt(
        self,
        post_text: str,
        analysis: dict,
        style: str,
        short_post: bool,
        retry: bool = False,
    ) -> str:

        retry_instruction = ""

        if retry:
            retry_instruction = """
IMPORTANT:
The previous reply failed validation.

For this attempt:
- Stay extremely close to the exact wording of the source post.
- Do not introduce new technical concepts.
- Do not assume implementation details.
- Do not invent results, mechanisms, causes, benchmarks, settings,
  architecture details, authentication methods, or deployment details.
- If you are asking a question, every technical concept in the question
  must be explicitly present in the source post or be an obvious
  grammatical reference to something present in it.
- Prefer an observation over a strong factual claim.
- If the source makes a contested claim, discuss the distinction or
  implication rather than asserting that claim as fact.
"""

        short_post_instruction = ""

        if short_post:
            short_post_instruction = """
SHORT POST MODE:
The source post is very short.

Because there is little context available:
- Use the exact topic, phrase, entity, or claim from the post.
- Do not expand it into an imagined technical explanation.
- Do not introduce specific mechanisms that are not explicitly mentioned.
- A simple observation or clarification question is preferred.
- If the post is mostly a title, react to the title itself.
- If the source is only a paper title/link, do not infer the paper's findings, methods,
  results, or conclusions from the title. Comment only on what the title explicitly states.
"""

        analysis_context = self._analysis_context(
            analysis
        )

        style_instruction = self._style_instruction(
            style
        )

        return f"""
You are generating one short reply for Bluesky.

Your job is to participate in the conversation naturally,
not to advertise, farm engagement, or sound like an AI interviewer.

SOURCE POST:
{post_text}

{self._source_grounding_guide(post_text)}

BOT NICHE:
{self.niche}

POST ANALYSIS:
{analysis_context}

REPLY STYLE:
{style_instruction}

{short_post_instruction}

{retry_instruction}

GENERAL RULES:

1. Reply directly to the source post.

1A. POLITICAL / POLICY NEUTRALITY:
    If the source concerns elections, politicians, parties, legislation,
    government policy, public-policy disputes, or geopolitics:
    - Do not endorse or oppose a candidate, party, policy, law, or political position.
    - Do not tell people how to vote.
    - Do not rank candidates, parties, policies, or choices.
    - Do not predict who will win an election.
    - Do not repeat a political claim as independently established fact when
      the source presents it as a claim, opinion, proposal, or disputed figure.
    - Prefer neutral observations about the stated claim, tradeoff, evidence,
      uncertainty, or question raised by the post.
    - If asking a question, ask about the issue actually raised rather than
      prompting political persuasion.

2. Use only information supported by the source post.
   If the source does not contain enough factual detail for a useful reply, output exactly: SKIP
   Never fill missing context from your own knowledge or from a linked article that you have not been given. Never invent implementation details, mechanisms, recommendations, causes, benefits, or implications.
Prefer replies that stay at the same factual level as the source:
- Restate or connect two explicit source facts when that adds value.
- Do not turn a list of facts into a claim about a trend, motivation, benefit, risk, importance, or broader implication.
- Do not use words such as "underscoring", "signaling", "showing", "highlighting", "suggesting", "widening", "improving", "streamlining", "reducing", or "enabling" unless the source explicitly supports that relationship.
- If the source is only a headline, title, announcement, or short list, keep the reply equally concrete and do not infer missing details.

   A URL, title, hashtag, or named product does not authorize assumptions about its implementation, results, mechanisms, or benefits.

   SOURCE-DETAIL MODE:
   Prefer selecting, combining, or lightly rephrasing concrete details already present in the source.
   Do not turn a source detail into a new judgment, implication, recommendation, mechanism, benefit, cause, or consequence.
   If you cannot produce a natural reply using only source-supported details, output exactly: SKIP.

3. You may restate or lightly interpret what the author said,
   but do not invent technical details.

3B. ADD VALUE, DO NOT JUST PARAPHRASE:
    The reply should contribute at least one small useful observation,
    implication, distinction, or interpretation when the source provides
    enough information to support it. Do not simply rewrite the source post
    in different words. A concise paraphrase is acceptable only for a very
    short source where there is not enough information for a deeper point.

3A. CLAIM STRENGTH:
    Keep the same level of certainty as the source.
    - If the source says "could", "may", "suggests", or asks a question,
      do not rewrite it as "will", "proves", or an established result.
    - Do not add causal explanations that the source does not state.
    - Do not claim a mechanism merely because it sounds technically plausible.
    - Prefer "the post describes", "the result suggests", or a direct
      observation when the source does not establish a stronger conclusion.

3C. LINK / ARTICLE BOUNDARY:
    The Bluesky post is the only source you may use.
    If it contains a URL, title, or link to an article, paper, thread,
    or external page, you MUST NOT use facts, mechanisms, findings,
    recommendations, or implementation details that are only available
    on that linked page. The linked content has not been read by you.
    React only to what is actually present in the Bluesky post.

4. Do NOT assume:
   - implementation details
   - architecture
   - benchmarks
   - latency numbers
   - deployment environment
   - infrastructure
   - datasets
   - training methods
   - model settings
   - authentication
   - authorization
   - causes
   - mechanisms
   - performance improvements
   unless the source post explicitly supports them.

5. IMPORTANT QUESTION RULE:
   If you ask a question, the question must be grounded in the source.

   Do NOT ask:
   "How do the agents authenticate?"
   if authentication is not mentioned.

   Do NOT ask:
   "What database are you using?"
   if a database is not mentioned.

   Do NOT ask:
   "How did you optimize latency?"
   if latency or optimization is not mentioned.

   Questions should explore something the author actually mentioned.

6. If the post makes a claim, do not automatically turn that claim
   into an established fact.

7. For contested, uncertain, or opinion-based claims, prefer wording like:
   "The distinction here is interesting..."
   "That raises an interesting question..."
   "The tradeoff seems to be..."
   "I wonder whether..."
   "The interesting part is..."
   "That framing makes X worth considering..."

8. Do not invent facts just to make the reply sound technical.

9. Avoid generic filler such as:
   "This is fascinating!"
   "Very interesting!"
   "Great post!"
   "This is huge!"

10. Do not ask for likes, reposts, follows, engagement, or shares.

11. Do not mention being an AI or bot.

12. Do not use hashtags unless absolutely necessary.

13. Do not use emojis.

14. Do not repeat the entire source post.

15. Avoid sounding like an interviewer.
    Not every reply should be a question.

16. Prefer a concrete observation when the source already contains a result,
    benchmark, architecture detail, paper title, or specific technical claim.
    A question is optional, not a default.

17. Do not attribute motives or intentions unless the author explicitly states them.
    For example, do not write "exactly why you built this" unless the source
    explicitly gives that reason.

15A. DO NOT REPEAT PROMOTIONAL CALLS TO ACTION.
    Avoid generic praise such as "Great demo", "Nice work", "Interesting",
    or "Love this" unless it is necessary to the point. Start with the
    concrete detail from the source instead.
    If the source contains a sign-up, registration, download, demo, purchase,
    or similar call to action, do not repeat that CTA. Discuss the underlying
    idea or product claim instead, without promoting it.

16. SOUND LIKE A PERSON, NOT A COMMENTARY TEMPLATE.
     Avoid canned openers such as:
     "Interesting—..."
     "The X part is interesting..."
     "The X angle is interesting..."
     "The post highlights..."
     "This highlights..."
     "This underscores..."
     "This raises..."
     "That framing..."
     "The key is..."
     "The distinction here is..."
     Do not start with generic praise. Start with the actual detail you are
     responding to, or connect it naturally to the post. Vary sentence
     structure across replies. The result should feel like a quick human
     reaction, not a summary followed by praise or an interview question.

17. Keep the reply between {self.MIN_WORDS} and {self.MAX_WORDS} words.
    Aim for {self.MIN_WORDS}–{self.TARGET_MAX_WORDS} words whenever possible.
    Prefer one clear idea over multiple clauses.

18. If asking about something the author found, reported, or discovered,
    do not assume they personally implemented or fixed it. Use wording
    such as "What would you change..." or "What stands out..." when appropriate.

19. Prefer observations or concise insights when they make a complete point.
Do not end every reply with a question. Ask a question only when it opens a
specific discussion that the source genuinely supports.

19A. VALUE TEST:
    Before writing, identify the smallest useful thing you can add to the
    conversation. This can be a concrete implication, distinction, limitation,
    comparison, or technical consequence explicitly supported by the source.
    If you cannot add one without speculation, keep the reply short and close
    to the source rather than inventing context.

20. QUESTION FREQUENCY: Most replies should NOT contain a question.
A question is optional, not a default ending. Never turn a complete observation
into a question merely to invite engagement.

21. PLATFORM-SAFE WORDING: Avoid the exact phrase "system prompt" in the
reply when the same source-supported idea can be expressed as "prompt",
"policy", "instructions", or another wording explicitly supported by the post.
Do not invent technical mechanisms while rephrasing.
Output ONLY the reply text.
"""

    # ------------------------------------------------------------------
    # STYLE
    # ------------------------------------------------------------------

    def _style_instruction(self, style: str) -> str:

        styles = {

            "observation": """
Make a concise, conversational observation about one concrete detail
already present in the post.

Do NOT ask a question or end with a question mark. Make the observation
feel complete on its own.

Do not use a canned "interesting" opener.

Loose examples of structure:
- "The shift from X to Y changes what the system has to handle."
- "The part about X gives the idea a much clearer shape."
- "The X result is the detail I'd keep an eye on."

Use these only as loose patterns. Do not copy them mechanically.
Do not introduce a new technical claim.
""",

            "technical_question": """
Ask one technically relevant question about something explicitly
mentioned in the post.

The question must NOT introduce a new technical concept.
Every technical noun or mechanism in the question must be explicitly present
in the source post or in the source-supported technical whitelist above.
Do not invent combinations such as "dynamic policy updates" or
"secure inter-agent communication" unless those ideas are actually stated.

Make it sound like a real follow-up from someone who read the post.
Prefer a concrete observation over a question when the source already
contains enough information to make a useful point. If a question is used,
make it specific to something the source actually mentions.

The question should refer directly to an entity, mechanism,
tradeoff, result, architecture, tool, or claim already present.

Bad:
"How do the agents authenticate?"
when authentication was not mentioned.

Good:
"What tradeoff did you see between the two approaches?"
when two approaches are explicitly discussed.
""",

            "follow_up": """
Continue one idea already present in the post.

Use a question ONLY when the source clearly opens a specific discussion.
Otherwise write a complete non-question observation. Never add a question
just to increase engagement.

Do not introduce a new technical concept.

Do not force a question if an observation would work better.
""",

            "concise_insight": """
Give a short insight based directly on the post.

Do NOT ask a question or end with a question mark. Make the insight
feel complete on its own.

Do not present speculation as fact.

If the author makes a strong or debatable claim,
comment on the distinction, implication, or tradeoff
rather than automatically agreeing with it.

For political or policy claims, remain descriptive and neutral.
""",

            "counterpoint": """
Offer a mild, constructive qualification.

Do not be argumentative for the sake of engagement.

Only challenge or qualify something that is actually stated
or clearly implied by the source post.

Use cautious language rather than declaring the author wrong.
""",
        }

        return styles.get(
            style,
            styles["observation"],
        )

    # ------------------------------------------------------------------
    # ANALYSIS CONTEXT
    # ------------------------------------------------------------------

    def _analysis_context(
        self,
        analysis: dict,
    ) -> str:

        if not analysis:
            return "No additional analysis available."

        useful_keys = [
            "relevant",
            "relevance_score",
            "content_value_score",
            "discussion_score",
            "credibility_score",
            "spam_score",
            "risk_level",
            "action",
            "reply_idea",
        ]

        lines = []

        for key in useful_keys:

            if key in analysis:
                lines.append(
                    f"{key}: {analysis[key]}"
                )

        if not lines:
            return "No additional analysis available."

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # STYLE SELECTION
    # ------------------------------------------------------------------

    def choose_style(self, post_text, analysis=None, attempt=1):
        """Choose a reply style from the source instead of defaulting to questions."""
        text = (post_text or "").strip()
        lower = text.lower()
        analysis = analysis or {}

        # Policy-sensitive topics stay descriptive and neutral.
        political_markers = getattr(self, "POLITICAL_MARKERS", ())
        if any(marker in lower for marker in political_markers):
            styles = ["observation", "concise_insight", "follow_up"]
            return styles[(max(attempt, 1) - 1) % len(styles)]

        # Explicit questions should generally receive a useful answer or
        # follow-up rather than an unrelated observation.
        has_question = "?" in text
        question_words = re.search(
            r"\b(?:how|why|what|which|where|when|whether|could|can|does|do|is|are)\b",
            lower,
        )

        # Posts reporting measured results or comparisons are often better
        # served by an observation/insight than a forced question.
        result_signals = (
            "benchmark", "benchmarks", "result", "results", "shows",
            "found", "finding", "study", "paper", "evaluates",
            "evaluation", "tested", "testing", "experiment", "compared",
            "comparison", "outperforms", "improves", "degrades",
        )
        technical_signals = (
            "api", "sdk", "mcp", "llm", "model", "agent", "agents",
            "architecture", "inference", "memory", "context", "security",
            "protocol", "workflow", "tool", "tools", "cache", "token",
        )

        has_results = any(signal in lower for signal in result_signals)
        has_technical = any(signal in lower for signal in technical_signals)

        # Avoid a question-first pattern. Cycle the first attempts through
        # observation and insight, and reserve technical questions for posts
        # that actually contain a question/comparison/result signal.
        if has_question:
            styles = ["concise_insight", "observation", "follow_up"]
        elif question_words:
            styles = ["concise_insight", "observation", "follow_up"]
        elif has_results:
            styles = ["observation", "concise_insight", "follow_up"]
        elif has_technical:
            styles = ["observation", "concise_insight", "follow_up"]
        else:
            styles = ["observation", "concise_insight", "follow_up"]

        # Only permit a technical question when the source gives us a reason
        # to ask one. This prevents unrelated implementation questions.
        if has_question and has_technical:
            styles = styles + ["technical_question"]

        return styles[(max(attempt, 1) - 1) % len(styles)]

    def clean_reply(self, reply: str) -> str:

        if reply is None:
            return ""

        reply = str(reply).strip()

        # Remove markdown fences.
        reply = re.sub(
            r"^```(?:text)?\s*",
            "",
            reply,
            flags=re.IGNORECASE,
        )

        reply = re.sub(
            r"\s*```$",
            "",
            reply,
            flags=re.IGNORECASE,
        )

        # Remove common model prefixes.
        reply = re.sub(
            r"^(?:reply|response)\s*:\s*",
            "",
            reply,
            flags=re.IGNORECASE,
        )

        # Remove surrounding quotes.
        if len(reply) >= 2:

            if (
                reply[0] == '"'
                and reply[-1] == '"'
            ) or (
                reply[0] == "'"
                and reply[-1] == "'"
            ):

                reply = reply[1:-1].strip()

        # Normalize whitespace.
        reply = re.sub(
            r"\s+",
            " ",
            reply,
        ).strip()

        return reply

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_reply(
        self,
        reply: str,
        post_text: str,
        analysis: dict | None = None,
        short_post: bool = False,
    ) -> tuple[bool, str]:

        if not reply:
            return False, "Empty reply"

        word_count = len(
            re.findall(
                r"\b[\w'-]+\b",
                reply,
            )
        )

        if word_count < self.MIN_WORDS:
            return (
                False,
                f"Too short ({word_count} words)",
            )

        if word_count > self.MAX_WORDS:
            return (
                False,
                f"Too long ({word_count} words)",
            )

        # Sentence completeness.
        if self.looks_incomplete(reply):
            return False, "Incomplete sentence"

        # Engagement bait.
        if self.contains_engagement_bait(reply):
            return False, "Contains engagement bait"

        # Generic filler.
        if self.is_generic_filler(reply):
            return False, "Generic filler"

        # Hard reject the recurring low-information templates that can survive
        # lexical grounding merely because they mention a source keyword.
        generic_observation_patterns = [
            r"^the\s+\w+(?:\s+\w+){0,5}\s+detail\s+is\s+what\s+makes\s+this\s+worth\s+(?:exploring|digging into)$",
            r"^the\s+\w+(?:\s+\w+){0,5}\s+(?:angle|piece|result|behavior|integration)\s+(?:is|gives|makes)\b",
            r"^the\s+(?:agent|report|inference|security|memory|reasoning|benchmark)\s+(?:behavior|detail|piece|result|angle)\s+is\s+the\s+part\b",
            r"^the\s+distinction\s+here\s+is\s+worth\s+exploring\b",
            r"^the\s+report\s+detail\s+is\s+what\s+makes\s+this\s+worth\s+exploring\.?$",
        ]
        if any(re.search(pattern, reply.strip(), re.I) for pattern in generic_observation_patterns):
            return False, "Low-information generic observation"

        # Naturalness / anti-template check.
        natural, natural_reason = self.check_naturalness(reply)
        if not natural:
            return False, natural_reason

        # Source grounding.
        grounded, grounding_reason = (
            self.check_grounding(
                reply=reply,
                post_text=post_text,
                short_post=short_post,
            )
        )

        if not grounded:
            return False, grounding_reason

        # Technical grounding.
        technical_ok, technical_reason = (
            self.check_technical_grounding(
                reply=reply,
                post_text=post_text,
            )
        )

        if not technical_ok:
            return False, technical_reason

        # Questions get an additional strict check.
        if "?" in reply:

            question_ok, question_reason = (
                self.check_question_grounding(
                    reply=reply,
                    post_text=post_text,
                )
            )

            if not question_ok:
                return False, question_reason

        marketing_ok, marketing_reason = self.check_marketing_language(
            reply=reply,
            post_text=post_text,
        )
        if not marketing_ok:
            return False, marketing_reason

        # Political/policy neutrality.
        political_ok, political_reason = self.check_political_neutrality(
            reply=reply,
            post_text=post_text,
        )

        if not political_ok:
            return False, political_reason

        # Claim strength.
        claim_ok, claim_reason = (
            self.check_claim_strength(
                reply=reply,
                post_text=post_text,
            )
        )

        if not claim_ok:
            return False, claim_reason

        # Do not approve a reply that is essentially a paraphrase of the
        # source without adding a small supported observation or implication.
        value_ok, value_reason = self.check_added_value(
            reply=reply,
            post_text=post_text,
            short_post=short_post,
        )

        if not value_ok:
            return False, value_reason

        # Repetition.
        if self.is_repetitive_with_post(
            reply,
            post_text,
        ):
            return (
                False,
                "Reply mostly repeats source post",
            )

        # Detect common robotic templates.
        if self.contains_repetitive_template(reply):
            return (
                False,
                "Reply uses repetitive template",
            )

        return True, (
            "Passed all reply validation checks"
        )

    # ------------------------------------------------------------------
    # SHORT POST
    # ------------------------------------------------------------------

    @staticmethod
    def is_short_post(post_text: str) -> bool:

        words = re.findall(
            r"\b[\w'-]+\b",
            post_text,
        )

        return len(words) <= 12

    # ------------------------------------------------------------------
    # INCOMPLETE SENTENCE
    # ------------------------------------------------------------------

    @staticmethod
    def looks_incomplete(text: str) -> bool:

        stripped = text.strip()

        if not stripped:
            return True

        if re.search(
            r"(?:\b(?:and|or|but|because|so|with|to|of|for|the|a|an|in|on|at|by|from|than|as))[\s.,!?]*$",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True

        if stripped.endswith(
            (
                "...",
                "—",
                "-",
            )
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # ENGAGEMENT BAIT
    # ------------------------------------------------------------------

    @staticmethod
    def contains_engagement_bait(text: str) -> bool:

        text_lower = text.lower()

        patterns = [
            r"\blike this\b",
            r"\blike the post\b",
            r"\bgive this a like\b",
            r"\blike if\b",
            r"\brepost this\b",
            r"\bshare this\b",
            r"\bfollow me\b",
            r"\bfollow us\b",
            r"\bfollow for\b",
            r"\bcheck my profile\b",
            r"\bboost this\b",
            r"\bshare your thoughts\b",
            r"\bwhat do you think\b",
            r"\bagree\?\b",
            r"\bthoughts\?\b",
        ]

        return any(
            re.search(
                pattern,
                text_lower,
            )
            for pattern in patterns
        )

    # ------------------------------------------------------------------
    # GENERIC FILLER
    # ------------------------------------------------------------------

    @staticmethod
    def is_generic_filler(text: str) -> bool:

        normalized = re.sub(
            r"[^a-z0-9\s]",
            "",
            text.lower(),
        )

        filler_patterns = [
            "this is fascinating",
            "this is very interesting",
            "very interesting post",
            "great post",
            "great work",
            "interesting stuff",
            "this is huge",
            "love this",
            "really interesting",
            "so interesting",
        ]

        return normalized.strip() in filler_patterns

    # ------------------------------------------------------------------
    # NATURALNESS
    # ------------------------------------------------------------------

    @classmethod
    def check_naturalness(cls, text: str) -> tuple[bool, str]:
        """Reject canned AI-commentary openings and stock phrasing."""
        normalized = re.sub(r"\\s+", " ", (text or "").strip().lower())

        if not normalized:
            return False, "Empty reply"

        for pattern in cls.ROBOTIC_OPENING_PATTERNS:
            if re.search(pattern, normalized, re.I):
                return False, "Reply starts with a robotic/template phrase"

        generic_hits = sum(
            1
            for pattern in cls.GENERIC_NATURALNESS_PATTERNS
            if re.search(pattern, normalized, re.I)
        )
        if generic_hits >= 2:
            return False, "Reply contains too much generic commentary"

        return True, "Naturalness acceptable"

    # ------------------------------------------------------------------
    # REPETITIVE TEMPLATE DETECTION
    # ------------------------------------------------------------------

    @staticmethod
    def contains_repetitive_template(
        text: str,
    ) -> bool:

        normalized = text.lower().strip()

        patterns = [
            r"^the .+ part is especially interesting",
            r"^the .+ angle is especially interesting",
            r"^the .+ part is interesting",
            r"^the .+ angle is interesting",
            r"^interesting[—,-]\s*how",
            r"^interesting[—,-]\s*what",
            r"^this is especially interesting",
            r"^this makes .+ interesting",
            r"^the post highlights\b",
            r"^the article highlights\b",
            r"^this highlights\b",
            r"^this underscores\b",
            r"^this raises\b",
            r"^that framing\b",
            r"^the key is to\b",
            r"^the distinction here is\b",
        ]

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in patterns
        )

    # ------------------------------------------------------------------
    # GROUNDING
    # ------------------------------------------------------------------

    def check_grounding(
        self,
        reply: str,
        post_text: str,
        short_post: bool = False,
    ) -> tuple[bool, str]:

        post_lower = post_text.lower()
        reply_lower = reply.lower()

        post_tokens = self.tokenize(
            post_lower
        )

        reply_tokens = self.tokenize(
            reply_lower
        )

        if not post_tokens:
            return (
                False,
                "Source post has no usable concepts",
            )

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
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
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "this",
            "that",
            "these",
            "those",
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
            "who",
            "does",
            "do",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
        }

        meaningful_post = {
            token
            for token in post_tokens
            if token not in stop_words
            and len(token) >= 3
        }

        meaningful_reply = {
            token
            for token in reply_tokens
            if token not in stop_words
            and len(token) >= 3
        }

        if not meaningful_reply:
            return (
                False,
                "Reply has no meaningful concepts",
            )

        overlap = (
            meaningful_post
            & meaningful_reply
        )

        compound_overlap = (
            self.find_compound_overlap(
                reply,
                post_text,
            )
        )

        if compound_overlap:
            return (
                True,
                "Grounded through named/compound concept",
            )

        if short_post:

            if len(overlap) >= 1:
                return (
                    True,
                    "Grounded in short source post",
                )

            return (
                False,
                "No meaningful concept overlap with short post",
            )

        if len(overlap) >= 2:
            return True, "Grounded"

        strong_terms = {
            token
            for token in overlap
            if token in self.TECHNICAL_TERMS
        }

        if strong_terms:
            return (
                True,
                "Grounded through technical concept",
            )

        uncommon_overlap = {
            token
            for token in overlap
            if len(token) >= 6
        }

        if uncommon_overlap:
            return (
                True,
                "Grounded through source terminology",
            )

        return (
            False,
            "No meaningful concept overlap",
        )

    # ------------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> set[str]:

        tokens = re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]*",
            text.lower(),
        )

        return set(tokens)

    # ------------------------------------------------------------------
    # COMPOUND OVERLAP
    # ------------------------------------------------------------------

    def find_compound_overlap(
        self,
        reply: str,
        post_text: str,
    ) -> bool:

        post_lower = post_text.lower()
        reply_lower = reply.lower()

        compounds = re.findall(
            r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b",
            post_lower,
        )

        for compound in compounds:

            pieces = [
                piece
                for piece in re.split(
                    r"[-_]",
                    compound,
                )
                if piece
            ]

            if compound in reply_lower:
                return True

            if len(pieces) >= 2:

                present = sum(
                    1
                    for piece in pieces
                    if re.search(
                        rf"\b{re.escape(piece)}\b",
                        reply_lower,
                    )
                )

                if present >= max(
                    2,
                    len(pieces) - 1,
                ):
                    return True

        quoted_phrases = re.findall(
            r'["“](.{3,80})["”]',
            post_text,
        )

        for phrase in quoted_phrases:

            phrase_lower = (
                phrase.lower().strip()
            )

            if phrase_lower in reply_lower:
                return True

        return False

    # ------------------------------------------------------------------
    # TECHNICAL GROUNDING
    # ------------------------------------------------------------------

    def check_technical_grounding(
        self,
        reply: str,
        post_text: str,
    ) -> tuple[bool, str]:
        """
        Enforce strict source-grounding for technical vocabulary.

        A technical concept is allowed only when the source explicitly contains
        that concept (including a simple singular/plural or inflectional variant).
        We intentionally do NOT use broad technical-family matching here.

        Example:
            Source mentions ``agents`` -> ``agent`` is allowed.
            Source mentions ``tools`` -> ``tool-calling`` is NOT allowed.
            Source mentions ``retrieval`` -> ``RAG`` is NOT automatically allowed.
            Source mentions ``RCE`` -> ``remote code execution`` is allowed only
            through the explicit acronym expansion below.
        """

        reply_lower = (reply or "").lower()
        post_lower = (post_text or "").lower()

        reply_terms = self.extract_technical_terms(reply_lower)
        post_terms = self.extract_technical_terms(post_lower)

        # Explicit technical-detail phrases are intentionally conservative.
        # If a model adds one of these concrete mechanisms/details, the exact
        # concept must occur in the source post. This prevents plausible
        # outside-knowledge expansions such as "zero-trust", "tool-calling",
        # "voice commands", or "IoT" from slipping through.
        concrete_detail_patterns = [
            r"\bzero[- ]trust\b",
            r"\btool[- ]calling\b",
            r"\banomaly detection\b",
            r"\bvoice commands?\b",
            r"\biot\b",
            r"\brbac\b",
            r"\bowasp\b",
            r"\bremote code execution\b",
            r"\bsmart[- ]home controller\b",
        ]

        for pattern in concrete_detail_patterns:
            if not re.search(pattern, reply_lower, re.I):
                continue
            if not re.search(pattern, post_lower, re.I):
                return (
                    False,
                    "Reply introduces unsupported technical detail",
                )

        if not reply_terms:
            return True, "No unsupported technical terminology"

        unsupported = []

        # Only very conservative acronym/expansion pairs are permitted.
        # These represent the same named concept rather than a related concept.
        equivalent_terms = {
            "rce": {"remote code execution"},
            "remote code execution": {"rce"},
        }

        for term in reply_terms:
            if term in post_terms or self.term_variant_in_set(term, post_terms):
                continue

            # Allow a reply term when the exact concept is present as a
            # hyphenated source compound, e.g. source ``LLM-based`` -> ``LLM``.
            source_compound_support = False
            for source_term in post_terms:
                normalized_source = re.sub(
                    r"[^a-z0-9]+", "-", source_term
                ).strip("-")
                source_parts = {
                    part for part in normalized_source.split("-") if part
                }
                if term in source_parts:
                    source_compound_support = True
                    break

            if source_compound_support:
                continue

            # Do not infer related concepts from a technical family.
            # ``agents`` does not automatically authorize ``tool-calling``;
            # ``retrieval`` does not automatically authorize ``RAG``; etc.

            # Generic AI nouns are safe when the source clearly establishes
            # an AI/agent/model context. They are intentionally limited to
            # generic nouns rather than mechanisms or architectures.
            if term in {"ai", "agent", "agents", "model", "models"}:
                if any(
                    re.search(
                        rf"\b{re.escape(marker)}\b",
                        post_lower,
                    )
                    for marker in (
                        "ai", "agent", "agents", "llm", "llms",
                        "model", "models",
                    )
                ):
                    continue

            # Explicit same-concept acronym expansion.
            equivalents = equivalent_terms.get(term, set())
            if equivalents:
                for equivalent in equivalents:
                    if re.search(
                        rf"(?<![\w-]){re.escape(equivalent)}(?![\w-])",
                        post_lower,
                    ):
                        continue
                else:
                    unsupported.append(term)
                    continue

            unsupported.append(term)

        if unsupported:
            return (
                False,
                "Unsupported technical terms: "
                + ", ".join(sorted(set(unsupported))),
            )

        return True, "Technical terminology strictly grounded in source"


    # ------------------------------------------------------------------
    # SIMPLE TERM VARIANT MATCHING
    # ------------------------------------------------------------------

    @staticmethod
    def term_variant_in_set(term, allowed_terms):
        term = (term or "").lower().strip()
        if not term:
            return False

        allowed_terms = {
            str(x).lower().strip()
            for x in (allowed_terms or set())
        }

        if term in allowed_terms:
            return True

        # Source compounds such as KV-cache, token-level and long-context
        # explicitly ground their component terms.
        normalized = re.sub(r"[^a-z0-9]+", "-", term).strip("-")
        components = {part for part in normalized.split("-") if part}
        if components & allowed_terms:
            return True

        irregular = {
            "llm": {"llms"},
            "llms": {"llm"},
            "model": {"models"},
            "models": {"model"},
            "evaluation": {"evaluations", "evaluate", "evaluated", "evaluating"},
            "evaluations": {"evaluation"},
            "benchmark": {"benchmarks", "benchmarking"},
            "benchmarks": {"benchmark"},
            "agent": {"agents"},
            "agents": {"agent"},
            "workflow": {"workflows"},
            "workflows": {"workflow"},
            "token": {"tokens"},
            "tokens": {"token"},
            "cache": {"caches", "caching"},
            "caches": {"cache"},
            "memory": {"memories"},
            "memories": {"memory"},
        }

        return bool(irregular.get(term, set()) & allowed_terms)

    def is_term_family_grounded(
        self,
        term: str,
        post_terms: set[str],
    ) -> bool:

        for family in self.TERM_FAMILIES:

            if term not in family:
                continue

            if family & post_terms:
                return True

        return False

    # ------------------------------------------------------------------
    # TECHNICAL TERM EXTRACTION
    # ------------------------------------------------------------------

    def extract_technical_terms(
        self,
        text: str,
    ) -> set[str]:

        terms = set()

        for term in self.TECHNICAL_TERMS:

            if re.search(
                rf"(?<![\w-])"
                rf"{re.escape(term)}"
                rf"(?![\w-])",
                text,
            ):
                terms.add(term)

        # Hashtags often concatenate technical terms, e.g.
        #InferenceOptimization. Treat a recognizable technical component
        #inside a hashtag as source-supported vocabulary.
        for tag in re.findall(r"#([A-Za-z][A-Za-z0-9_-]*)", text):
            tag_lower = tag.lower()
            normalized_tag = tag_lower.replace("-", "")
            for term in self.TECHNICAL_TERMS:
                if len(term) < 4:
                    continue
                normalized_term = term.lower().replace("-", "")
                if normalized_term in normalized_tag:
                    terms.add(term)

        compounds = re.findall(
            r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+\b",
            text,
        )

        for compound in compounds:

            if any(
                piece in self.TECHNICAL_TERMS
                for piece in compound.split("-")
            ):
                terms.add(compound)

        # Ambiguous words should only count as technical when the source
        # provides technical context. This prevents governance sandboxes
        # from being treated as software sandboxes, for example.
        if "sandbox" in terms or "sandboxed" in terms:
            if not re.search(
                r"\b(?:software|code|api|test|testing|isolat|runtime|agent|model|compute|security)\b",
                text,
                re.I,
            ):
                terms.discard("sandbox")
                terms.discard("sandboxed")

        if "rag" in terms and not re.search(
            r"\b(?:retrieval|vector|embedding|document|knowledge|search|context)\b",
            text,
            re.I,
        ):
            terms.discard("rag")

        return terms

    # ------------------------------------------------------------------
    # QUESTION GROUNDING
    # ------------------------------------------------------------------

    def check_question_grounding(
        self,
        reply: str,
        post_text: str,
    ) -> tuple[bool, str]:

        """
        Questions are stricter than normal observations.

        A question must not introduce a new technical concept.
        """

        reply_terms = (
            self.extract_technical_terms(
                reply.lower()
            )
        )

        post_terms = (
            self.extract_technical_terms(
                post_text.lower()
            )
        )

        unsupported = []

        for term in reply_terms:

            if term in post_terms or self.term_variant_in_set(term, post_terms):
                continue

            # Do not infer a technical concept merely because it belongs to
            # the same broad family as something in the source.
            # For example, ``retrieval`` does not authorize ``RAG`` and
            # ``tools`` does not authorize ``tool-calling``.

            # Generic AI terms are allowed when the post
            # clearly establishes an AI/agent/model context.
            if term in {
                "ai",
                "agent",
                "agents",
                "model",
                "models",
            }:
                if any(
                    marker in post_terms
                    for marker in {
                        "ai",
                        "agent",
                        "agents",
                        "llm",
                        "llms",
                    }
                ):
                    continue

            unsupported.append(term)

        if unsupported:
            return (
                False,
                "Question introduces unsupported "
                "technical concepts: "
                + ", ".join(
                    sorted(unsupported)
                ),
            )

        # Also catch common technical concepts that aren't
        # in our term dictionary but are clearly specific.
        #
        # These are intentionally conservative.
        suspicious_concepts = {
            "authentication",
            "authorization",
            "database",
            "databases",
            "latency",
            "throughput",
            "benchmark",
            "benchmarks",
            "evaluation",
            "evals",
            "deployment",
            "deployments",
            "infrastructure",
            "architecture",
            "security",
            "protocol",
            "protocols",
            "api",
            "apis",
            "rpc",
            "microservice",
            "microservices",
        "tokenizer",
        "tokenizers",
        "softmax",
        "confidence-threshold",
        "confidence-thresholds",
        "parallelism",
        "distributed",
        "distribution",
        "docs",
        "documentation",
        "tool-output",
        "tool-outputs",
        "feedback-loop",
        "feedback-loops",
        "editorial",
        "editorial-voice",
        "mimic",
        "mimics",
        "recent",
        "human",
            "policy",
            "policies",
            "communication",
            "communications",
            "control plane",
            "inter-agent",
        }

        post_lower = post_text.lower()
        reply_lower = reply.lower()

        for concept in suspicious_concepts:

            concept_pattern = (
                rf"\b{re.escape(concept)}\b"
                if " " not in concept
                else rf"(?<!\w){re.escape(concept)}(?!\w)"
            )

            if not re.search(concept_pattern, reply_lower):
                continue

            if re.search(concept_pattern, post_lower):
                continue

            # Family support.
            concept_terms = {
                term
                for term in post_terms
                if self.is_term_family_grounded(
                    concept,
                    {term},
                )
            }

            if concept_terms:
                continue

            return (
                False,
                "Question introduces unsupported "
                f"technical concept: {concept}",
            )

        return (
            True,
            "Question is grounded in source",
        )

    # ------------------------------------------------------------------
    # POLITICAL / POLICY NEUTRALITY
    # ------------------------------------------------------------------

    @classmethod
    def is_political_or_policy_post(cls, post_text: str) -> bool:
        text = (post_text or "").lower()
        return any(
            re.search(rf"(?<![\\w-]){re.escape(marker)}(?![\\w-])", text)
            for marker in cls.POLITICAL_MARKERS
        )

    @classmethod
    def check_political_neutrality(
        cls,
        reply: str,
        post_text: str,
    ) -> tuple[bool, str]:
        if not cls.is_political_or_policy_post(post_text):
            return True, "Not a political/policy post"

        if re.search(r"\b(?:the\s+)?key\s+is\s+to\b", reply, re.I):
            return False, "Reply gives prescriptive political/policy guidance"

        reply_lower = reply.lower()

        for pattern in cls.POLITICAL_ENDORSEMENT_PATTERNS:
            if re.search(pattern, reply_lower):
                return False, "Reply contains political endorsement, opposition, or voting guidance"

        # Avoid turning the source's normative language into the bot's own
        # conclusion. A reply can quote/refer to the source, but otherwise
        # should use neutral framing.
        normative = re.search(
            r"\b(should|must|need to|have to|ought to)\b",
            reply_lower,
        )
        if normative:
            if not any(
                re.search(pattern, reply_lower)
                for pattern in cls.POLITICAL_NEUTRAL_PATTERNS
            ):
                return False, "Reply uses normative language on a political/policy post"

        # Election outcome claims are especially risky.
        if re.search(
            r"\b(will|would|is likely to|is more likely to)\s+win\b",
            reply_lower,
        ):
            return False, "Reply predicts an election outcome"

        return True, "Political/policy neutrality acceptable"

    # ------------------------------------------------------------------
    # CLAIM STRENGTH
    # ------------------------------------------------------------------

    @classmethod
    def check_marketing_language(cls, reply: str, post_text: str):
        lower = (reply or "").lower().strip()
        for pattern in cls.MARKETING_PATTERNS:
            if re.search(pattern, lower, re.I):
                return False, "Reply contains promotional or marketing language"

        source = (post_text or "").lower()
        promotional_source = any(
            re.search(p, source, re.I)
            for p in [
                r"\bstreamline\b",
                r"\bsign up\b",
                r"\blearn more\b",
                r"\bbook a demo\b",
                r"\brequest a demo\b",
                r"\bdeal[- ]sourcing\b",
                r"\bavailable at\b",
                r"\bcheck out\b",
            ]
        )
        if promotional_source and re.search(
            r"\b(?:best|great|excellent|amazing|fantastic|powerful)\s+(?:product|platform|tool|service|solution)\b",
            lower,
        ):
            return False, "Reply praises or promotes a promotional source"
        return True, "No promotional language detected"


        # Reject common "fact -> broader implication" constructions unless the
        # source explicitly contains the same relationship. These phrases are
        # prone to turning a headline/list of facts into a new claim about
        # trends, motives, benefits, risks, or significance.
        post_lower = (post_text or "").lower()
        reply_lower = (reply or "").lower()

        interpretation_patterns = [
            r"\b(?:underscor(?:es|ing)|highlight(?:s|ing)|signal(?:s|ing)|signif(?:y|ies|ying))\b",
            r"\b(?:show(?:s|ing)?|demonstrat(?:es|ing)|indicat(?:es|ing))\b",
            r"\b(?:suggest(?:s|ing)|imply(?:ies|ing)|reflect(?:s|ing))\b",
            r"\b(?:widen(?:s|ing)?|expand(?:s|ing)?|broad(?:ens|ening))\b.{0,40}\b(?:attack surface|risk|scope|reach|exposure)\b",
            r"\b(?:growing|increasing|rising|greater|broader|stronger)\s+(?:focus|interest|attention|adoption|demand|risk|concern)\b",
            r"\b(?:streamlin(?:es|ing)|reduc(?:es|ing)|improv(?:es|ing)|enhanc(?:es|ing)|enabl(?:es|ing)|simplif(?:ies|ying))\b",
            r"\b(?:safeguard(?:s|ing)|protect(?:s|ing)|mitigat(?:es|ing)|prevent(?:s|ing))\b",
        ]

        reply_has_interpretation = any(
            re.search(pattern, reply_lower)
            for pattern in interpretation_patterns
        )
        source_has_interpretation = any(
            re.search(pattern, post_lower)
            for pattern in interpretation_patterns
        )

        if reply_has_interpretation and not source_has_interpretation:
            return (
                False,
                "Reply adds an unsupported broader interpretation or implication",
            )


        # Reject recommendations, necessity claims, and inferred importance
        # when the source does not explicitly make the same claim.
        prescriptive_patterns = [
            r"\b(?:essential|necessary|critical|important|key|crucial)\b",
            r"\b(?:isn't|is not|aren't|are not)\s+enough\b",
            r"\b(?:must|need to|needs to|should|have to|has to)\b",
            r"\b(?:validate|validates|validation)\b",
            r"\b(?:proves?|demonstrates?)\b",
            r"\b(?:ensures?|guarantees?)\b",
            r"\b(?:trust alone)\b",
        ]

        reply_has_prescriptive_language = any(
            re.search(pattern, reply_lower)
            for pattern in prescriptive_patterns
        )
        source_has_prescriptive_language = any(
            re.search(pattern, post_lower)
            for pattern in prescriptive_patterns
        )

        if reply_has_prescriptive_language and not source_has_prescriptive_language:
            return (
                False,
                "Reply adds an unsupported recommendation, necessity, or importance claim",
            )

    def check_claim_strength(
        self,
        reply: str,
        post_text: str,
    ) -> tuple[bool, str]:

        reply_lower = reply.lower()
        post_lower = post_text.lower()

        # A question is not normally a factual assertion.
        if reply.strip().endswith("?"):
            return (
                True,
                "Question does not overstate source claim",
            )

        claim_matches = []

        for word in self.CLAIM_WORDS:

            if re.search(
                rf"\b{re.escape(word)}\b",
                reply_lower,
            ):
                claim_matches.append(word)

        if not claim_matches:
            return (
                True,
                "No strong factual claim detected",
            )

        strong_claim = any(
            re.search(
                pattern,
                reply_lower,
            )
            for pattern in self.STRONG_CLAIM_PATTERNS
        )

        if strong_claim:

            post_tokens = self.tokenize(
                post_lower
            )

            reply_tokens = self.tokenize(
                reply_lower
            )

            meaningful_post = {
                token
                for token in post_tokens
                if len(token) >= 4
            }

            meaningful_reply = {
                token
                for token in reply_tokens
                if len(token) >= 4
            }

            overlap = (
                meaningful_post
                & meaningful_reply
            )

            if len(overlap) < 4:

                return (
                    False,
                    "Claim is stronger than the source supports",
                )

        # Reject vague interpretive framing even when the nouns themselves
        # are grounded. Prefer concrete source details over evaluation.
        interpretive_framing_patterns = [
            r"\bgives\s+(?:the|this|that|an?)\s+(?:idea|approach|concept|post)\s+(?:a\s+)?(?:clearer|clear|better)\s+shape\b",
            r"\bshows\s+why\b",
            r"\bhighlights?\s+(?:the|how|why)\b",
            r"\b(?:clear|useful|strong|interesting)\s+operational\s+hook\b",
            r"\b(?:makes|gives|provides)\s+(?:a\s+)?(?:clear|useful|good)\s+(?:sense|view|picture)\b",
            r"\b(?:signals?|suggests?)\s+(?:a|an|the)\b",
        ]
        if any(re.search(pattern, reply_lower) for pattern in interpretive_framing_patterns):
            return False, "Reply adds unsupported interpretive framing"

        # Strong certainty / benefit language must be explicitly supported.
        # Do not let the model upgrade "can help" / "without X" into
        # "eliminates", "guarantees", "ensures", etc.
        certainty_patterns = [
            r"\b(?:eliminates?|guarantees?|ensures?|solves?|removes?|prevents?|avoids?|requires?\s+no)\b",
            r"\b(?:removes?|eliminates?)\s+(?:the|any|all)\s+need\s+for\b",
        ]
        reply_has_strong_certainty = any(
            re.search(pattern, reply_lower)
            for pattern in certainty_patterns
        )
        source_has_strong_certainty = any(
            re.search(pattern, post_lower)
            for pattern in certainty_patterns
        )
        if reply_has_strong_certainty and not source_has_strong_certainty:
            return (
                False,
                "Reply upgrades the source into an unsupported certainty or benefit claim",
            )

        # Operational/consequence phrases must be explicitly grounded.
        # These are easy places for the model to add plausible but unsupported
        # implications (for example, "without manual review").
        operational_concepts = {
            "manual review",
            "human review",
            "human oversight",
            "without review",
            "without manual",
            "production",
            "deployable",
            "deployment",
            "automatically",
            "automation",
        }

        for concept in operational_concepts:
            concept_pattern = (
                rf"(?<!\w){re.escape(concept)}(?!\w)"
            )
            if not re.search(concept_pattern, reply_lower):
                continue
            if not re.search(concept_pattern, post_lower):
                return (
                    False,
                    f"Reply introduces unsupported operational concept: {concept}",
                )

        # Consequence / mechanism language is a common hallucination path.
        # If the source does not explicitly state the relationship, reject it.
        consequence_patterns = [
            r"\b(?:can|could|may|will|would)\s+(?:increase|decrease|reduce|improve|worsen|limit|prevent|cause|lead|enable|allow|require)\b",
            r"\b(?:increases?|decreases?|reduces?|improves?|worsens?|limits?|prevents?|causes?|leads?|enables?|allows?|requires?)\b",
            r"\b(?:adds?|introduces?|creates?)\s+(?:a|an|the)\s+(?:step|layer|constraint|overhead|risk|benefit|tradeoff|requirement)\b",
            r"\b(?:therefore|thus|hence|which means|meaning|as a result|resulting in)\b",
            r"\bpoints? to (?:a|an|the)\b",
            r"\bmiss(?:es|ing)?\b",
            r"\bgap\b",
        ]

        source_has_consequence_language = any(
            re.search(pattern, post_lower)
            for pattern in consequence_patterns
        )
        reply_has_consequence_language = any(
            re.search(pattern, reply_lower)
            for pattern in consequence_patterns
        )

        if reply_has_consequence_language and not source_has_consequence_language:
            return (
                False,
                "Reply introduces an unsupported consequence or causal relationship",
            )

        # Causal language requires explicit source support.
        has_causal_language = any(
            re.search(
                pattern,
                reply_lower,
            )
            for pattern in self.CAUSAL_PATTERNS
        )

        if has_causal_language:

            source_causal_words = [
                "because",
                "due to",
                "causes",
                "caused",
                "prevents",
                "prevent",
                "leads",
                "leading",
                "results",
                "result",
                "therefore",
            ]

            if not any(
                word in post_lower
                for word in source_causal_words
            ):

                return (
                    False,
                    "Reply introduces unsupported causality",
                )

        # Explicit certainty about a source claim should be
        # rejected unless the source itself clearly supports it.
        certainty_patterns = [
            r"\bso\s+\w+\s+is\b",
            r"\btherefore\b",
            r"\bthis\s+means\b",
            r"\bin\s+fact\b",
            r"\bindeed\b",
            r"\bactually\b",
        ]

        if any(
            re.search(
                pattern,
                reply_lower,
            )
            for pattern in certainty_patterns
        ):

            source_tokens = self.tokenize(
                post_lower
            )

            reply_tokens = self.tokenize(
                reply_lower
            )

            overlap = (
                source_tokens
                & reply_tokens
            )

            if len(overlap) < 3:

                return (
                    False,
                    "Reply asserts an unsupported conclusion",
                )

        return (
            True,
            "Claim strength acceptable",
        )

    # ------------------------------------------------------------------
    # ADDED VALUE
    # ------------------------------------------------------------------

    @classmethod
    def check_added_value(
        cls,
        reply: str,
        post_text: str,
        short_post: bool = False,
    ) -> tuple[bool, str]:
        """Reject source paraphrases that contribute no useful observation."""
        if short_post:
            return True, "Short source; paraphrase allowed"

        reply_tokens = cls.tokenize(reply)
        post_tokens = cls.tokenize(post_text)

        meaningful_reply = {t for t in reply_tokens if len(t) >= 4}
        meaningful_post = {t for t in post_tokens if len(t) >= 4}

        if len(meaningful_reply) < 4 or len(meaningful_post) < 6:
            return True, "Not enough vocabulary for a reliable value check"

        overlap = len(meaningful_reply & meaningful_post) / max(len(meaningful_reply), 1)

        # These phrases usually signal that the writer is adding an implication,
        # distinction, limitation, comparison, or consequence rather than merely
        # repeating the source. They do not by themselves make a claim factual.
        value_markers = [
            r"\bwhich means\b",
            r"\bmeaning\b",
            r"\bsuggests?\b",
            r"\bimplies?\b",
            r"\bpoints? to\b",
            r"\bhelps? explain\b",
            r"\bmakes it\b",
            r"\bnot just\b",
            r"\brather than\b",
            r"\bwhereas\b",
            r"\bthe implication\b",
            r"\bthe limitation\b",
            r"\bthe constraint\b",
            r"\bthe tradeoff\b",
            r"\bthe difference\b",
            r"\bthe consequence\b",
            r"\bthe failure mode\b",
            r"\buseful because\b",
            r"\bimportant because\b",
            r"\bespecially when\b",
            r"\beasier to\b",
            r"\bharder to\b",
        ]

        has_value_marker = any(
            re.search(pattern, reply.lower())
            for pattern in value_markers
        )

        # High lexical overlap + no interpretive signal is a strong indication
        # that the reply is only restating the post. Keep questions exempt because
        # their grounded-question validator handles their contribution separately.
        if overlap >= 0.72 and "?" not in reply and not has_value_marker:
            return False, "Reply mostly paraphrases the source without adding value"

        return True, "Reply adds sufficient value or is not a close paraphrase"

    # ------------------------------------------------------------------
    # REPETITION WITH SOURCE
    # ------------------------------------------------------------------

    def is_repetitive_with_post(
        self,
        reply: str,
        post_text: str,
    ) -> bool:

        reply_tokens = self.tokenize(
            reply
        )

        post_tokens = self.tokenize(
            post_text
        )

        if not reply_tokens or not post_tokens:
            return False

        meaningful_reply = {
            token
            for token in reply_tokens
            if len(token) >= 4
        }

        meaningful_post = {
            token
            for token in post_tokens
            if len(token) >= 4
        }

        if not meaningful_reply:
            return False

        overlap = (
            meaningful_reply
            & meaningful_post
        )

        ratio = len(overlap) / len(
            meaningful_reply
        )

        return (
            ratio >= 0.85
            and len(meaningful_reply) >= 5
        )


