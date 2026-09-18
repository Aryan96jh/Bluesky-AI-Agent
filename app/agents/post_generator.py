"""Generates original standalone posts (not replies) based on what's trending."""

from __future__ import annotations

from typing import Optional


class PostGenerator:
    """Writes original Bluesky posts about AI/crypto from hype-ranked topics.

    Bluesky caps a post at 300 characters, so generated posts are kept
    short but substantive -- a full single post, not a one-liner.
    """

    MAX_POST_CHARS = 280  # stay safely under Bluesky's 300-char limit

    def __init__(
        self,
        llm_client=None,
        niche: str = "",
        tone: str = "",
    ):
        self.llm = llm_client
        self.niche = niche or "AI agents, LLMs and crypto"
        self.tone = tone or "casual, knowledgeable, slightly witty"

    def generate(
        self,
        trending: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict:
        """Generate one original post from trending topics.

        trending: list of {"text": str, "author": str, "hype": int}
        Returns {"text": str, "topic": str} or {"error": str}.
        """
        if self.llm is None:
            return {"error": "No LLM client configured"}

        if not trending:
            return {"error": "No trending topics provided"}

        context_lines = []
        for item in trending[:5]:
            text = str(item.get("text", "")).replace("\n", " ").strip()
            if len(text) > 160:
                text = text[:157] + "..."
            context_lines.append(
                f"- (hype {item.get('hype', 0)}) @{item.get('author', '?')}: {text}"
            )
        context = "\n".join(context_lines)

        prompt = (
            f"You are a {self.niche} commentator on Bluesky. "
            f"Tone: {self.tone}.\n\n"
            "These topics are trending right now in the AI/crypto space:\n"
            f"{context}\n\n"
            "Write ONE original standalone post (not a reply) about AI or crypto, "
            "inspired by what's trending above. Rules:\n"
            f"- Maximum {self.MAX_POST_CHARS} characters (hard limit, Bluesky allows 300)\n"
            "- Substantive and insightful, not a generic one-liner\n"
            "- No more than 1 hashtag, no emoji spam (0-1 emoji max)\n"
            "- Do not start with 'Breaking', 'Hot take', or 'Unpopular opinion'\n"
            "- Do not ask the reader a question just to farm engagement\n"
            "- Sound like a human expert, not a marketing bot\n\n"
            "Return ONLY the post text, nothing else."
        )

        try:
            text = self.llm.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            return {"error": f"LLM generation failed: {exc}"}

        text = (text or "").strip().strip('"').strip()

        if not text:
            return {"error": "LLM returned empty post"}

        if len(text) > 300:
            # Hard cut at Bluesky's limit, preferring a word boundary.
            cut = text[:297]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            text = cut

        topic_hint = str(trending[0].get("text", ""))[:80]

        return {"text": text, "topic": topic_hint}

    def validate(self, text: str) -> tuple[bool, str]:
        """Lightweight sanity checks before publishing."""
        if not text or not text.strip():
            return False, "empty post"
        if len(text) > 300:
            return False, f"post too long ({len(text)} chars > 300)"
        if len(text) < 40:
            return False, "post too short to be substantive"
        lowered = text.lower()
        for spam in ("buy now", "click the link", "dm me", "guaranteed profit"):
            if spam in lowered:
                return False, f"spam phrase detected: {spam!r}"
        return True, "ok"
