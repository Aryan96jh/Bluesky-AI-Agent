import os
import re
import time
from datetime import datetime, timezone

from app.agents.analyzer import TweetAnalyzer
from app.agents.ai_validator import AIAnalysisValidator
from app.agents.hype import HypeDetector
from app.agents.policy import ActionPolicy
from app.agents.topic_generator import TopicGenerator
from app.agents.reply_generator import ReplyGenerator
from app.bluesky.client import BlueskyClient
from app.config import BOT_NICHE, BOT_TONE
from app.memory.database import BotMemory
from app.memory.smart_memory import SmartMemory
from app.llm.groq_client import GroqClient
from app.agents.action_queue import ActionQueue
from app.agents.rate_limiter import RateLimiter
from app.agents.promotion_detector import PromotionDetector
from app.agents.post_generator import PostGenerator


class BotRunner:

    def __init__(self):

        self.bluesky = BlueskyClient()

        self.memory = BotMemory()
        self.smart_memory = SmartMemory()

        self.llm = GroqClient()

        self.analyzer = TweetAnalyzer(
            llm_client=self.llm,
            niche=BOT_NICHE,
            tone=BOT_TONE,
        )

        self.reply_generator = ReplyGenerator(
            llm_client=self.llm,
            niche=BOT_NICHE,
            tone=BOT_TONE,
        )

        self.ai_validator = AIAnalysisValidator(
            min_discovery_score=float(
                os.getenv(
                    "AI_VALIDATION_DISCOVERY_THRESHOLD",
                    "60",
                )
            ),
            min_technical_score=float(
                os.getenv(
                    "AI_VALIDATION_TECHNICAL_THRESHOLD",
                    "65",
                )
            ),
            min_evidence_score=float(
                os.getenv(
                    "AI_VALIDATION_EVIDENCE_THRESHOLD",
                    "35",
                )
            ),
            max_contradictory_relevance=float(
                os.getenv(
                    "AI_VALIDATION_MAX_LOW_RELEVANCE",
                    "20",
                )
            ),
        )

        self.hype = HypeDetector()

        self.topic_generator = TopicGenerator()

        self.policy = ActionPolicy(
            min_relevance=60,
            min_opportunity=30,
            min_credibility=20,
            max_spam=60,
        )

        self.dry_run = (
            os.getenv(
                "DRY_RUN",
                "true",
            ).lower()
            == "true"
        )

        self.max_candidates = int(
            os.getenv(
                "MAX_CANDIDATES",
                "12",
            )
        )

        self.max_ai_analyses = int(
            os.getenv(
                "MAX_AI_ANALYSES",
                "5",
            )
        )

        self.max_replies_per_day = int(
            os.getenv(
                "MAX_REPLIES_PER_DAY",
                "5",
            )
        )

        self.search_limit_per_topic = int(
            os.getenv(
                "SEARCH_LIMIT_PER_TOPIC",
                "25",
            )
        )

        self.search_hours = int(
            os.getenv(
                "SEARCH_HOURS",
                "168",
            )
        )

        self.min_hype_score = int(
            os.getenv(
                "MIN_HYPE_SCORE",
                "0",
            )
        )

        # Minimum engagement (likes + replies*3 + reposts*4) for a post
        # to be considered. Filters out dead posts with no traction so
        # the bot engages with slightly popular, reachable conversations.
        self.min_engagement = int(
            os.getenv(
                "MIN_ENGAGEMENT",
                "5",
            )
        )

        self.min_discovery_score = float(
            os.getenv(
                "MIN_DISCOVERY_SCORE",
                "35",
            )
        )

        self.delay_between_actions = float(
            os.getenv(
                "ACTION_DELAY_SECONDS",
                "15",
            )
        )

        self.max_live_actions_per_run = int(
            os.getenv(
                "MAX_LIVE_ACTIONS_PER_RUN",
                "1",
            )
        )

        # Two independent safety gates are required before LIVE execution:
        # 1. DRY_RUN must be false.
        # 2. LIVE_EXECUTION_ENABLED must be true.
        self.live_execution_enabled = (
            os.getenv(
                "LIVE_EXECUTION_ENABLED",
                "false",
            ).lower()
            == "true"
        )

        self.promotion_detector = PromotionDetector(
            threshold=int(
                os.getenv(
                    "PROMOTION_DETECTOR_THRESHOLD",
                    "5",
                )
            )
        )

        self.action_queue = ActionQueue()
        self.rate_limiter = RateLimiter(
            memory=self.memory,
            max_actions_per_day=self.max_replies_per_day,
            delay_seconds=self.delay_between_actions,
            max_actions_per_run=self.max_live_actions_per_run,
            live_execution_enabled=(
                not self.dry_run
                and self.live_execution_enabled
            ),
        )

        # -------------------------------------------------
        # ORIGINAL POSTS (standalone posts, not replies)
        # -------------------------------------------------
        self.post_generator = PostGenerator(
            llm_client=self.llm,
            niche=BOT_NICHE,
            tone=BOT_TONE,
        )

        self.enable_original_posts = (
            os.getenv(
                "ENABLE_ORIGINAL_POSTS",
                "false",
            ).lower()
            == "true"
        )

        self.max_original_posts_per_run = int(
            os.getenv(
                "MAX_ORIGINAL_POSTS_PER_RUN",
                "1",
            )
        )

        self.max_original_posts_per_day = int(
            os.getenv(
                "MAX_ORIGINAL_POSTS_PER_DAY",
                "1",
            )
        )

    # =========================================================
    # MAIN RUN
    # =========================================================

    def run(self):

        print()
        print(
            "=================================================="
        )
        print(
            "NICHEBOT STARTING"
        )
        print(
            "=================================================="
        )

        stats_before = self.memory.get_stats()

        print(
            f"[MEMORY] "
            f"Total={stats_before['total']} "
            f"Analyzed={stats_before['analyzed']} "
            f"Unanalyzed={stats_before['unanalyzed']}"
        )

        smart_stats = self.smart_memory.stats()
        print(
            f"[SMART MEMORY] Authors={smart_stats['authors']} "
            f"Interactions={smart_stats['interactions']}"
        )

        print(
            f"[ACTIVITY] "
            f"Live replies today="
            f"{stats_before['replies_today']} "
            f"/ {self.max_replies_per_day}"
        )

        print(
            f"[ACTIVITY] "
            f"Dry-run replies today="
            f"{stats_before['dry_run_replies_today']}"
        )

        print(
            f"[MODE] "
            f"DRY_RUN={self.dry_run}"
        )

        print(
            f"[LIVE SAFETY] "
            f"LIVE_EXECUTION_ENABLED={self.live_execution_enabled}"
        )

        print(
            f"[RATE LIMIT] "
            f"Live actions/run={self.max_live_actions_per_run}, "
            f"live replies/day={self.max_replies_per_day}, "
            f"delay={self.delay_between_actions:.1f}s"
        )

        if not self.dry_run and not self.live_execution_enabled:
            print(
                "[LIVE SAFETY] LIVE mode requested, but the explicit "
                "LIVE_EXECUTION_ENABLED gate is OFF."
            )
            print(
                "[LIVE SAFETY] No actions will be published."
            )

        print()

        candidates = self.discover_candidates()

        if not candidates:

            print(
                "[RUNNER] No eligible candidates found."
            )

            self.print_summary(
                candidates=[],
                analyzed=0,
                approved=0,
                simulated=0,
                executed=0,
            )

            return

        print()
        print(
            f"[DISCOVERY] Found "
            f"{len(candidates)} candidates."
        )

        ai_analyses = 0
        approved = 0
        smart_skipped = 0
        promotion_skipped = 0
        simulated = 0
        executed = 0

        for index, post in enumerate(
            candidates,
            start=1,
        ):

            if ai_analyses >= self.max_ai_analyses:

                print()
                print(
                    "[RUNNER] AI analysis limit reached."
                )

                break

            print()
            print(
                "=================================================="
            )

            print(
                f"Candidate {index}:"
            )

            print(
                f"Author: @{post.author_handle}"
            )

            print(
                f"Post: {post.text}"
            )

            print(
                f"URL: {post.url}"
            )

            print(
                "=================================================="
            )

            # -------------------------------------------------
            # AI ANALYSIS
            # -------------------------------------------------

            print(
                "[AI] Analyzing..."
            )

            analysis = self.analyzer.analyze(
                post_text=post.text,
            )

            ai_analyses += 1

            if "error" in analysis:

                print(
                    "[AI] Analysis failed:"
                )

                print(
                    analysis["error"]
                )

                print(
                    "[MEMORY] Analysis was NOT marked complete."
                )

                continue

            # -------------------------------------------------
            # AI SANITY / VALIDATION
            # -------------------------------------------------

            discovery_data = (
                self.get_discovery_data(
                    post
                )
            )

            validation = (
                self.ai_validator.validate(
                    analysis=analysis,
                    discovery=discovery_data,
                    post_text=post.text,
                )
            )

            if not validation["valid"]:

                print()
                print(
                    "[AI VALIDATION] FAILED"
                )

                print(
                    f"  Discovery score: "
                    f"{validation.get('discovery_score', 0):.1f}"
                )

                print(
                    f"  Technical score: "
                    f"{validation.get('technical_score', 0):.1f}"
                )

                print(
                    f"  Evidence score: "
                    f"{validation.get('evidence_score', 0):.1f}"
                )

                print(
                    f"  AI relevance: "
                    f"{validation.get('ai_relevance', 0):.1f}"
                )

                anchors = validation.get(
                    "anchors",
                    [],
                )

                if anchors:

                    print(
                        f"  Niche anchors: "
                        f"{', '.join(anchors)}"
                    )

                for reason in validation["reasons"]:

                    print(
                        f"  - {reason}"
                    )

                print(
                    "[MEMORY] Analysis was NOT marked complete."
                )

                print(
                    "[AI VALIDATION] "
                    "Post will remain eligible for a future run."
                )

                continue

            print()
            print(
                "[AI VALIDATION] Passed."
            )

            # -------------------------------------------------
            # -------------------------------------------------
            # PROMOTION / SELF-PROMO DETECTOR
            # -------------------------------------------------

            promotion_result = self.promotion_detector.analyze(
                post.text
            )

            print()
            print("[PROMOTION DETECTOR]")
            print(
                f"  Score: {promotion_result.score}/10"
            )
            print(
                f"  Promotional: "
                f"{promotion_result.is_promotional}"
            )

            for reason in promotion_result.reasons:
                print(
                    f"  - {reason}"
                )

            if promotion_result.is_promotional:
                print(
                    "[PROMOTION DETECTOR] "
                    "Promotional post detected."
                )
                print(
                    "[PROMOTION DETECTOR] "
                    "Action changed to ignore."
                )

                analysis["action"] = "ignore"
                analysis["reply_idea"] = ""
                analysis["reply_style"] = ""
                analysis["reason"] = (
                    "Promotion detector flagged post: "
                    + "; ".join(
                        promotion_result.reasons
                    )
                )

                promotion_skipped += 1

            # -------------------------------------------------
            # DEDICATED REPLY GENERATOR
            # -------------------------------------------------
            # -------------------------------------------------

            if analysis.get("action") == "reply":

                print()
                print(
                    "[REPLY GENERATOR] Generating final reply..."
                )

                generated_reply = self.reply_generator.generate(
                    post_text=post.text,
                    analysis=analysis,
                )

                if generated_reply:

                    analysis["reply_idea"] = generated_reply

                    print(
                        f'[REPLY GENERATOR] Reply: "{generated_reply}"'
                    )

                else:

                    analysis["reply_idea"] = ""

                    print(
                        "[REPLY GENERATOR] Failed to generate reply."
                    )

                    print(
                        "[REPLY GENERATOR] Action changed to ignore."
                    )

                    analysis["action"] = "ignore"

                    analysis["reply_style"] = ""

                    analysis["reason"] = (
                        "Dedicated reply generator failed"
                    )

            # -------------------------------------------------
            # FINAL REPLY SAFETY CHECK
            # -------------------------------------------------

            if analysis.get("action") == "reply":

                final_reply = (
                    analysis.get(
                        "reply_idea",
                        "",
                    )
                    or ""
                ).strip()

                reply_validation = self.validate_final_reply(
                    post.text,
                    final_reply,
                )

                print()
                print(
                    "[REPLY SAFETY]"
                )

                print(
                    f"  Valid: {reply_validation['valid']}"
                )

                for reason in reply_validation["reasons"]:

                    print(
                        f"  - {reason}"
                    )

                if not reply_validation["valid"]:

                    print(
                        "[REPLY SAFETY] Rejected. "
                        "Action changed to ignore."
                    )

                    analysis["action"] = "ignore"
                    analysis["reply_idea"] = ""
                    analysis["reply_style"] = ""
                    analysis["reason"] = (
                        "Final reply failed safety validation: "
                        + "; ".join(reply_validation["reasons"])
                    )

            # -------------------------------------------------
            # SMART MEMORY: TOPIC / REPLY DUPLICATE CHECK
            # -------------------------------------------------

            final_reply = (
                analysis.get(
                    "reply_idea",
                    "",
                )
                or ""
            ).strip()

            if analysis.get("action") == "reply":

                topic_match = self.smart_memory.topic_overlap(
                    post.text,
                    days=int(
                        os.getenv(
                            "TOPIC_MEMORY_DAYS",
                            "14",
                        )
                    ),
                    threshold=float(
                        os.getenv(
                            "TOPIC_MEMORY_THRESHOLD",
                            "0.35",
                        )
                    ),
                )

                if topic_match is not None:

                    print()
                    print(
                        "[SMART MEMORY] Similar topic was engaged recently."
                    )
                    print(
                        f"[SMART MEMORY] Similarity: "
                        f"{topic_match['similarity']:.2f}"
                    )
                    print(
                        "[SMART MEMORY] Action changed to ignore."
                    )

                    analysis["action"] = "ignore"
                    analysis["reply_idea"] = ""
                    analysis["reply_style"] = ""
                    analysis["reason"] = (
                        "Similar topic was already engaged recently"
                    )
                    smart_skipped += 1

                elif final_reply:

                    reply_match = self.smart_memory.similar_reply(
                        final_reply,
                        days=int(
                            os.getenv(
                                "REPLY_MEMORY_DAYS",
                                "30",
                            )
                        ),
                        threshold=float(
                            os.getenv(
                                "REPLY_MEMORY_THRESHOLD",
                                "0.60",
                            )
                        ),
                    )

                    if reply_match is not None:

                        print()
                        print(
                            "[SMART MEMORY] Similar reply already used."
                        )
                        print(
                            f"[SMART MEMORY] Similarity: "
                            f"{reply_match['similarity']:.2f}"
                        )
                        print(
                            "[SMART MEMORY] Action changed to ignore."
                        )

                        analysis["action"] = "ignore"
                        analysis["reply_idea"] = ""
                        analysis["reply_style"] = ""
                        analysis["reason"] = (
                            "Very similar reply was already used recently"
                        )
                        smart_skipped += 1

            # -------------------------------------------------
            # SAVE VALIDATED ANALYSIS
            # -------------------------------------------------

            self.memory.save_analysis(
                post.uri,
                analysis,
            )

            print(
                "[AI] Analysis:"
            )

            print(
                f"  Relevant: "
                f"{analysis.get('relevant')}"
            )

            print(
                f"  Relevance: "
                f"{analysis.get('relevance_score')}"
            )

            print(
                f"  Content value: "
                f"{analysis.get('content_value_score')}"
            )

            print(
                f"  Discussion: "
                f"{analysis.get('discussion_score')}"
            )

            print(
                f"  Credibility: "
                f"{analysis.get('credibility_score')}"
            )

            print(
                f"  Spam: "
                f"{analysis.get('spam_score')}"
            )

            print(
                f"  Risk: "
                f"{analysis.get('risk_level')}"
            )

            print(
                f"  Action: "
                f"{analysis.get('action')}"
            )

            reply = (
                analysis.get(
                    "reply_idea",
                    "",
                )
                or ""
            ).strip()

            if reply:

                print(
                    f'AI reply: "{reply}"'
                )

            else:

                print(
                    "AI reply: [none]"
                )

            # -------------------------------------------------
            # POLICY
            # -------------------------------------------------

            decision = self.policy.decide(
                analysis
            )

            print()
            print(
                "[POLICY]"
            )

            print(
                f"  Allowed: "
                f"{decision['allowed']}"
            )

            print(
                f"  Action: "
                f"{decision['action']}"
            )

            print(
                f"  Reason: "
                f"{decision['reason']}"
            )

            if not decision["allowed"]:

                continue

            # -------------------------------------------------
            # QUALITY
            # -------------------------------------------------

            quality_score, quality_reasons = (
                self.calculate_quality(
                    post,
                    analysis,
                )
            )

            print()
            print(
                f"[QUALITY] Score: "
                f"{quality_score}"
            )

            for reason in quality_reasons:

                print(
                    f"  - {reason}"
                )

            if quality_score < 70:

                print(
                    "[QUALITY] Rejected."
                )

                continue

            print(
                "[QUALITY] Approved."
            )

            approved += 1

            # -------------------------------------------------
            # ACTION QUEUE
            # -------------------------------------------------

            self.action_queue.enqueue(
                action=analysis.get("action", "reply"),
                post=post,
                text=reply,
                analysis=analysis,
            )

            print()
            print(
                f"[ACTION QUEUE] Enqueued reply "
                f"({len(self.action_queue)} pending)."
            )

        # -----------------------------------------------------
        # EXECUTE QUEUED ACTIONS
        # -----------------------------------------------------

        print()
        print(
            f"[ACTION QUEUE] Processing {len(self.action_queue)} queued action(s)..."
        )

        while len(self.action_queue) > 0:

            queued = self.action_queue.pop_next()

            if queued is None:
                break

            if queued.action != "reply":
                print(
                    f"[ACTION QUEUE] Skipping unsupported action: {queued.action}"
                )
                continue

            if self.dry_run:

                print()
                print("[DRY RUN] Would reply:")
                print(f'    "{queued.text}"')

                self.memory.record_activity(
                    uri=queued.post.uri,
                    action="reply",
                    mode="dry_run",
                )

                self.smart_memory.record_interaction(
                    author=queued.post.author_handle,
                    post_text=queued.post.text,
                    reply=queued.text,
                    mode="dry_run",
                    action="reply",
                )

                simulated += 1

                print("[ACTIVITY] Recorded as dry_run.")
                continue

            allowed, limit_reason = self.rate_limiter.can_execute()

            if not allowed:
                print()
                print(f"[RATE LIMIT] {limit_reason}")
                print(
                    "[RATE LIMIT] Remaining queued actions were "
                    "not executed."
                )
                break

            self.rate_limiter.wait_if_needed(dry_run=False)

            try:

                print()
                print("[ACTION] Posting queued reply...")

                self.bluesky.reply_to_post(
                    text=queued.text,
                    post=queued.post,
                )

                self.memory.record_activity(
                    uri=queued.post.uri,
                    action="reply",
                    mode="live",
                )

                self.smart_memory.record_interaction(
                    author=queued.post.author_handle,
                    post_text=queued.post.text,
                    reply=queued.text,
                    mode="live",
                    action="reply",
                )

                self.memory.mark_executed(
                    queued.post.uri
                )

                executed += 1
                self.rate_limiter.mark_executed()

                print("[ACTION] Reply posted successfully.")
                print("[ACTIVITY] Recorded as live.")
                print(
                    f"[RATE LIMIT] Live actions this run: "
                    f"{self.rate_limiter.run_live_actions}/"
                    f"{self.max_live_actions_per_run}"
                )

            except Exception as e:

                print("[ACTION] Failed:")
                print(str(e))
                print("[ACTIVITY] Failed action was NOT recorded as live.")

        posts_made = self.maybe_publish_original_post(candidates)

        self.print_summary(
            candidates=candidates,
            analyzed=ai_analyses,
            approved=approved,
            simulated=simulated,
            executed=executed,
            smart_skipped=smart_skipped,
            promotion_skipped=promotion_skipped,
            posts_made=posts_made,
        )

    # =========================================================
    # ORIGINAL POSTS
    # =========================================================

    def maybe_publish_original_post(self, candidates) -> int:
        """Optionally publish one original standalone post.

        Picks the highest-hype discovered topics, generates a post with
        the LLM, and publishes it through the same safety gates as replies
        (DRY_RUN=false + LIVE_EXECUTION_ENABLED=true required for live).
        Returns the number of posts published this run (0 or more).
        """
        if not self.enable_original_posts:
            print()
            print("[ORIGINAL POST] Disabled (ENABLE_ORIGINAL_POSTS != true).")
            return 0

        print()
        print("==================================================")
        print("ORIGINAL POST")
        print("==================================================")

        # --- per-run + daily limits -------------------------------------
        run_posts = 0

        try:
            posts_today = self.memory.get_activity_today(
                mode="live",
                action="post",
            )
        except Exception:
            posts_today = 0

        if posts_today >= self.max_original_posts_per_day:
            print(
                f"[ORIGINAL POST] Daily limit reached "
                f"({posts_today}/{self.max_original_posts_per_day}). Skipping."
            )
            return 0

        # --- build trending context from hype-ranked candidates ----------
        scored = []
        for post in candidates or []:
            try:
                hype = self.hype.calculate_score(
                    likes=post.likes,
                    replies=post.replies,
                    reposts=post.reposts,
                    created_at=post.created_at,
                )
            except Exception:
                hype = 0
            scored.append({"post": post, "hype": hype})

        scored.sort(key=lambda item: item["hype"], reverse=True)

        trending = [
            {
                "text": item["post"].text,
                "author": item["post"].author_handle,
                "hype": item["hype"],
            }
            for item in scored[:5]
        ]

        if not trending:
            print("[ORIGINAL POST] No trending topics available. Skipping.")
            return 0

        print(
            "[ORIGINAL POST] Top hype topics: "
            + ", ".join(
                f"@{t['author']} ({t['hype']})" for t in trending[:3]
            )
        )

        # --- generate ----------------------------------------------------
        print("[ORIGINAL POST] Generating post text...")
        result = self.post_generator.generate(trending)

        if "error" in result:
            print(f"[ORIGINAL POST] Generation failed: {result['error']}")
            return 0

        text = result["text"]
        print()
        print("[ORIGINAL POST] Draft:")
        print(text)
        print(f"[ORIGINAL POST] Length: {len(text)} chars")

        # --- validate ----------------------------------------------------
        valid, reason = self.post_generator.validate(text)
        if not valid:
            print(f"[ORIGINAL POST] Validation failed: {reason}")
            return 0

        try:
            promo_result = self.promotion_detector.analyze(text)
            if getattr(promo_result, "is_promotional", False):
                print(
                    "[ORIGINAL POST] Blocked by promotion detector "
                    f"(score {getattr(promo_result, 'score', '?')})."
                )
                return 0
        except Exception as exc:
            print(f"[ORIGINAL POST] Promotion check error (ignoring): {exc}")

        # --- safety gates -------------------------------------------------
        live_allowed = (
            not self.dry_run and self.live_execution_enabled
        )

        if not live_allowed:
            print("[ORIGINAL POST] DRY RUN -- not publishing.")
            print(f"[ORIGINAL POST] Would publish: {text!r}")
            try:
                self.memory.record_activity(
                    uri=f"draft:{datetime.now(timezone.utc).isoformat()}",
                    action="post",
                    mode="dry_run",
                )
            except Exception:
                pass
            return 0

        if run_posts >= self.max_original_posts_per_run:
            print("[ORIGINAL POST] Per-run limit reached. Skipping.")
            return 0

        # --- publish ------------------------------------------------------
        self.rate_limiter.wait_if_needed(dry_run=False)

        try:
            print("[ORIGINAL POST] Publishing...")
            response = self.bluesky.send_post(text=text)
            uri = getattr(response, "uri", None) or (
                f"post:{datetime.now(timezone.utc).isoformat()}"
            )

            self.memory.record_activity(
                uri=uri,
                action="post",
                mode="live",
            )
            try:
                self.smart_memory.record_interaction(
                    author=BOT_NICHE,
                    post_text=result.get("topic", ""),
                    reply=text,
                    mode="live",
                    action="post",
                )
            except Exception:
                pass

            print("[ORIGINAL POST] Published successfully.")
            return 1

        except Exception as exc:
            print("[ORIGINAL POST] Failed to publish:")
            print(str(exc))
            return 0

    # =========================================================
    # DISCOVERY
    # =========================================================

    def discover_candidates(self):

        topics = (
            self.topic_generator.generate_topics()
        )

        print(
            f"[DISCOVERY] Searching "
            f"{len(topics)} topics..."
        )

        discovered = {}

        for topic in topics:

            try:

                posts = self.bluesky.search_posts_window(
                    query=topic,
                    limit=self.search_limit_per_topic,
                    hours=self.search_hours,
                )

            except Exception as e:

                print()
                print(
                    f"[DISCOVERY] Search failed for "
                    f"'{topic}': {e}"
                )

                continue

            for post in posts:

                if self.memory.has_analyzed(
                    post.uri
                ):

                    continue

                # -------------------------------------------------
                # SMART MEMORY: AUTHOR COOLDOWN
                # -------------------------------------------------

                if self.smart_memory.should_skip_author(
                    post.author_handle,
                    cooldown_hours=int(
                        os.getenv(
                            "AUTHOR_REPLY_COOLDOWN_HOURS",
                            "72",
                        )
                    ),
                ):

                    continue

                if not post.text.strip():

                    continue

                filter_result = (
                    self.local_content_filter(
                        post.text
                    )
                )

                if not filter_result["allowed"]:

                    continue

                hype_score = (
                    self.hype.calculate_score(
                        likes=post.likes,
                        replies=post.replies,
                        reposts=post.reposts,
                        created_at=post.created_at,
                    )
                )

                if (
                    hype_score
                    < self.min_hype_score
                ):

                    continue

                # Skip dead posts: require a minimum level of engagement
                # so replies land on slightly popular, reachable threads.
                engagement = (
                    max(post.likes, 0)
                    + max(post.replies, 0) * 3
                    + max(post.reposts, 0) * 4
                )

                if engagement < self.min_engagement:

                    continue

                technical_score = (
                    self.calculate_technical_score(
                        post.text
                    )
                )

                evidence_score = (
                    self.calculate_evidence_score(
                        post.text
                    )
                )

                discussion_score = (
                    self.calculate_discovery_discussion_score(
                        post.text
                    )
                )

                freshness_score = (
                    self.calculate_freshness_score(
                        post.created_at
                    )
                )

                text_quality_score = (
                    self.calculate_text_quality_score(
                        post.text
                    )
                )

                discovery_score = (
                    technical_score * 0.35
                    + evidence_score * 0.15
                    + discussion_score * 0.10
                    + freshness_score * 0.10
                    + hype_score * 0.25
                    + text_quality_score * 0.05
                )

                discovery_score = round(
                    min(
                        max(
                            discovery_score,
                            0,
                        ),
                        100,
                    ),
                    1,
                )

                if (
                    discovery_score
                    < self.min_discovery_score
                ):

                    continue

                content_key = (
                    self.content_key(
                        post.text
                    )
                )

                candidate = {
                    "post": post,
                    "hype": hype_score,
                    "technical": technical_score,
                    "evidence": evidence_score,
                    "discussion": discussion_score,
                    "freshness": freshness_score,
                    "text_quality": text_quality_score,
                    "discovery": discovery_score,
                    "content_key": content_key,
                }

                # -------------------------------------------------
                # Duplicate content handling
                #
                # Different accounts can post the same article,
                # paper or announcement. Keep only the strongest
                # version of essentially identical content.
                # -------------------------------------------------

                existing = None

                for existing_key, existing_item in discovered.items():

                    if self.are_content_duplicates(
                        content_key,
                        existing_key,
                    ):

                        existing = existing_item

                        break

                if existing is not None:

                    if (
                        candidate["discovery"]
                        > existing["discovery"]
                    ):

                        old_uri = (
                            existing["post"].uri
                        )

                        discovered.pop(
                            existing["content_key"],
                            None,
                        )

                        discovered[
                            content_key
                        ] = candidate

                        print(
                            "[DISCOVERY] Replaced duplicate "
                            f"with stronger post: "
                            f"{old_uri} -> {post.uri}"
                        )

                    continue

                discovered[
                    content_key
                ] = candidate

                self.memory.save_discovered_post(
                    post
                )

        ranked = sorted(
            discovered.values(),
            key=lambda item: item["discovery"],
            reverse=True,
        )

        print()
        print(
            f"[DISCOVERY] Collected "
            f"{len(ranked)} unique eligible posts "
            f"before ranking."
        )

        print()
        print(
            "Top local candidates:"
        )

        for index, item in enumerate(
            ranked[:15],
            start=1,
        ):

            post = item["post"]

            preview = (
                post.text
                .replace(
                    "\n",
                    " ",
                )
                .strip()
            )

            if len(preview) > 100:

                preview = (
                    preview[:97]
                    + "..."
                )

            print(
                f"{index}. "
                f"[Discovery {item['discovery']:.1f}] "
                f"[Tech {item['technical']}] "
                f"[Evidence {item['evidence']}] "
                f"[Discussion {item['discussion']}] "
                f"[Fresh {item['freshness']}] "
                f"[Hype {item['hype']}] "
                f"@{post.author_handle}: "
                f"{preview}"
            )

        candidates = [
            item["post"]
            for item in ranked[
                :self.max_candidates
            ]
        ]

        return candidates

    # =========================================================
    # DISCOVERY DATA FOR AI VALIDATION
    # =========================================================

    def get_discovery_data(self, post):

        technical_score = (
            self.calculate_technical_score(
                post.text
            )
        )

        evidence_score = (
            self.calculate_evidence_score(
                post.text
            )
        )

        discussion_score = (
            self.calculate_discovery_discussion_score(
                post.text
            )
        )

        freshness_score = (
            self.calculate_freshness_score(
                post.created_at
            )
        )

        text_quality_score = (
            self.calculate_text_quality_score(
                post.text
            )
        )

        hype_score = (
            self.hype.calculate_score(
                likes=post.likes,
                replies=post.replies,
                reposts=post.reposts,
                created_at=post.created_at,
            )
        )

        discovery_score = (
            technical_score * 0.35
            + evidence_score * 0.15
            + discussion_score * 0.10
            + freshness_score * 0.10
            + hype_score * 0.25
            + text_quality_score * 0.05
        )

        discovery_score = round(
            min(
                max(
                    discovery_score,
                    0,
                ),
                100,
            ),
            1,
        )

        return {
            "discovery_score": discovery_score,
            "technical_score": technical_score,
            "evidence_score": evidence_score,
            "discussion_score": discussion_score,
            "freshness_score": freshness_score,
            "hype_score": hype_score,
            "text_quality_score": text_quality_score,
        }

    # =========================================================
    # LOCAL CONTENT FILTER
    # =========================================================

    def local_content_filter(
        self,
        text: str,
    ):

        normalized = self.normalize_text(
            text
        )

        if not normalized:

            return {
                "allowed": False,
                "reason": "empty",
            }

        without_urls = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            normalized,
        ).strip()

        if len(without_urls) < 20:

            return {
                "allowed": False,
                "reason": "link-only or too short",
            }

        political_terms = [
            "senate",
            "congress",
            "parliament",
            "election",
            "president",
            "minister",
            "democrat",
            "republican",
            "political",
            "politics",
            "government vote",
            "legislation",
            "legislative",
            "campaign",
        ]

        political_hits = sum(
            1
            for term in political_terms
            if term in normalized
        )

        technical_score = (
            self.calculate_technical_score(
                text
            )
        )

        if (
            political_hits >= 2
            and technical_score < 45
        ):

            return {
                "allowed": False,
                "reason": "political noise",
            }

        generic_news_terms = [
            "travel restrictions",
            "stock market",
            "breaking news",
            "latest news",
            "breaking:",
            "reportedly",
            "according to officials",
            "amid tensions",
        ]

        news_hits = sum(
            1
            for term in generic_news_terms
            if term in normalized
        )

        if (
            news_hits >= 1
            and technical_score < 50
        ):

            return {
                "allowed": False,
                "reason": "generic news",
            }

        promotional_terms = [
            "buy now",
            "sign up",
            "download now",
            "get started",
            "limited time",
            "special offer",
            "free trial",
            "use my code",
            "discount",
            "sale",
        ]

        promotional_hits = sum(
            1
            for term in promotional_terms
            if term in normalized
        )

        if (
            promotional_hits >= 1
            and technical_score < 55
        ):

            return {
                "allowed": False,
                "reason": "promotional",
            }

        return {
            "allowed": True,
            "reason": "passed",
        }

    # =========================================================
    # TECHNICAL SCORING
    # =========================================================

    def calculate_technical_score(
        self,
        text: str,
    ):

        normalized = self.normalize_text(
            text
        )

        score = 0

        core_terms = {
            "ai": 8,
            "llm": 12,
            "large language model": 14,
            "agent": 12,
            "agents": 12,
            "ai agent": 15,
            "autonomous agent": 15,
            "coding agent": 15,
            "agentic": 12,
            "mcp": 15,
            "model": 7,
            "inference": 12,
            "training": 10,
            "fine tuning": 12,
            "finetuning": 12,
            "reasoning": 10,
            "tool calling": 14,
            "function calling": 14,
            "context window": 14,
            "embedding": 10,
            "embeddings": 10,
            "rag": 14,
        }

        for term, points in core_terms.items():

            if term in normalized:

                score += points

        technical_terms = {
            "benchmark": 15,
            "benchmarks": 15,
            "evaluation": 15,
            "eval": 12,
            "experiment": 15,
            "experiments": 15,
            "dataset": 12,
            "datasets": 12,
            "architecture": 14,
            "implementation": 12,
            "throughput": 14,
            "latency": 14,
            "memory": 10,
            "gpu": 10,
            "gpus": 10,
            "token": 8,
            "tokens": 8,
            "accuracy": 10,
            "precision": 10,
            "recall": 10,
            "performance": 10,
            "cache": 10,
            "caching": 10,
            "retrieval": 10,
            "reasoning benchmark": 16,
            "open source": 12,
            "github": 12,
            "arxiv": 15,
            "paper": 12,
            "research": 12,
            "method": 10,
            "methodology": 14,
            "ablation": 15,
            "reproducible": 15,
            "production": 8,
            "deployment": 10,
            "deployments": 10,
            "api": 8,
            "sdk": 10,
            "framework": 10,
            "infrastructure": 12,
            "security": 10,
            "reliability": 12,
            "failure": 8,
            "failures": 8,
        }

        for term, points in technical_terms.items():

            if term in normalized:

                score += points

        percentages = re.findall(
            r"\b\d+(?:\.\d+)?\s*%",
            normalized,
        )

        if percentages:

            score += min(
                len(percentages) * 8,
                24,
            )

        multipliers = re.findall(
            r"\b\d+(?:\.\d+)?\s*x\b",
            normalized,
        )

        if multipliers:

            score += min(
                len(multipliers) * 10,
                20,
            )

        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            normalized,
        )

        if numbers:

            score += min(
                len(numbers) * 3,
                12,
            )

        if "?" in text:

            score += 4

        discussion_terms = [
            "why",
            "how",
            "what happens",
            "tradeoff",
            "trade-off",
            "compare",
            "comparison",
            "versus",
            "vs",
            "problem",
            "challenge",
            "limitation",
            "limitations",
            "failure",
            "fails",
        ]

        discussion_hits = sum(
            1
            for term in discussion_terms
            if term in normalized
        )

        score += min(
            discussion_hits * 4,
            16,
        )

        word_count = len(
            normalized.split()
        )

        if word_count < 8:

            score -= 25

        elif word_count < 15:

            score -= 10

        url_count = len(
            re.findall(
                r"https?://\S+|www\.\S+",
                text.lower(),
            )
        )

        if url_count >= 2:

            score -= 8

        return max(
            0,
            min(
                round(score),
                100,
            ),
        )

    # =========================================================
    # EVIDENCE SCORING
    # =========================================================

    def calculate_evidence_score(
        self,
        text: str,
    ):

        normalized = self.normalize_text(
            text
        )

        score = 0

        strong_signals = {
            "benchmark": 18,
            "benchmarks": 18,
            "experiment": 18,
            "experiments": 18,
            "measured": 15,
            "measurement": 15,
            "results": 15,
            "result": 12,
            "evaluation": 16,
            "eval": 14,
            "ablation": 18,
            "methodology": 16,
            "method": 10,
            "dataset": 14,
            "datasets": 14,
            "paper": 15,
            "arxiv": 18,
            "research": 12,
            "github": 12,
            "implementation": 12,
            "reproducible": 18,
            "reproduction": 14,
            "architecture": 10,
            "latency": 12,
            "throughput": 12,
            "accuracy": 10,
            "precision": 10,
            "recall": 10,
            "memory usage": 12,
            "token usage": 12,
        }

        for term, points in strong_signals.items():

            if term in normalized:

                score += points

        percentage_count = len(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*%",
                normalized,
            )
        )

        multiplier_count = len(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*x\b",
                normalized,
            )
        )

        number_count = len(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                normalized,
            )
        )

        score += min(
            percentage_count * 8,
            20,
        )

        score += min(
            multiplier_count * 8,
            16,
        )

        score += min(
            number_count * 2,
            10,
        )

        if (
            "according to" in normalized
            or "we found" in normalized
            or "we measured" in normalized
            or "our results" in normalized
            or "results show" in normalized
        ):

            score += 8

        if (
            "github.com" in normalized
            or "arxiv.org" in normalized
        ):

            score += 10

        return max(
            0,
            min(
                round(score),
                100,
            ),
        )

    # =========================================================
    # DISCUSSION SCORING
    # =========================================================

    def calculate_discovery_discussion_score(
        self,
        text: str,
    ):

        normalized = self.normalize_text(
            text
        )

        score = 20

        question_signals = [
            "why",
            "how",
            "what",
            "when",
            "where",
            "which",
            "could",
            "should",
            "does",
            "is it",
            "are we",
        ]

        discussion_signals = [
            "tradeoff",
            "trade-off",
            "challenge",
            "problem",
            "limitation",
            "limitations",
            "failure",
            "fails",
            "issue",
            "bottleneck",
            "debate",
            "comparison",
            "versus",
            "vs",
            "alternative",
            "approach",
            "architecture",
            "design",
            "reliability",
            "evaluation",
        ]

        question_hits = sum(
            1
            for term in question_signals
            if term in normalized
        )

        discussion_hits = sum(
            1
            for term in discussion_signals
            if term in normalized
        )

        score += min(
            question_hits * 7,
            28,
        )

        score += min(
            discussion_hits * 6,
            30,
        )

        if "?" in text:

            score += 10

        word_count = len(
            normalized.split()
        )

        if 20 <= word_count <= 120:

            score += 10

        elif word_count > 180:

            score -= 5

        return max(
            0,
            min(
                round(score),
                100,
            ),
        )

    # =========================================================
    # FRESHNESS SCORING
    # =========================================================

    def calculate_freshness_score(
        self,
        created_at,
    ):

        if created_at is None:

            return 0

        now = datetime.now(
            timezone.utc
        )

        if created_at.tzinfo is None:

            created_at = created_at.replace(
                tzinfo=timezone.utc
            )

        age_hours = max(
            (
                now - created_at
            ).total_seconds()
            / 3600,
            0,
        )

        if age_hours <= 1:

            return 100

        if age_hours <= 3:

            return 95

        if age_hours <= 6:

            return 85

        if age_hours <= 12:

            return 72

        if age_hours <= 18:

            return 60

        if age_hours <= 24:

            return 50

        if age_hours <= 48:

            return 30

        return 10

    # =========================================================
    # TEXT QUALITY
    # =========================================================

    def calculate_text_quality_score(
        self,
        text: str,
    ):

        normalized = self.normalize_text(
            text
        )

        if not normalized:

            return 0

        score = 50

        words = normalized.split()

        word_count = len(
            words
        )

        if 15 <= word_count <= 120:

            score += 20

        elif 8 <= word_count < 15:

            score += 5

        elif word_count < 8:

            score -= 25

        elif word_count > 200:

            score -= 10

        sentence_count = len(
            re.findall(
                r"[.!?]+",
                text,
            )
        )

        if sentence_count >= 2:

            score += 10

        if sentence_count >= 4:

            score += 5

        url_count = len(
            re.findall(
                r"https?://\S+|www\.\S+",
                text.lower(),
            )
        )

        if url_count == 1:

            score += 3

        elif url_count >= 2:

            score -= 12

        promotional_terms = [
            "buy now",
            "sign up",
            "use my code",
            "limited time",
            "discount",
            "sale",
            "free trial",
        ]

        promotional_hits = sum(
            1
            for term in promotional_terms
            if term in normalized
        )

        score -= promotional_hits * 15

        return max(
            0,
            min(
                round(score),
                100,
            ),
        )

    # =========================================================
    # CONTENT KEY
    # =========================================================

    @staticmethod
    def content_key(
        text: str,
    ):

        normalized = text.lower()

        normalized = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"@[a-zA-Z0-9._-]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"[^\w\s]",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        tokens = normalized.split()

        # Keep enough content to recognize reposted articles
        # while avoiding tiny variations in punctuation.
        tokens = tokens[:80]

        return " ".join(
            tokens
        )

    # =========================================================
    # DUPLICATE DETECTION
    # =========================================================

    @staticmethod
    def are_content_duplicates(
        key_a: str,
        key_b: str,
    ):

        if not key_a or not key_b:

            return False

        if key_a == key_b:

            return True

        tokens_a = set(
            key_a.split()
        )

        tokens_b = set(
            key_b.split()
        )

        if not tokens_a or not tokens_b:

            return False

        intersection = (
            tokens_a
            & tokens_b
        )

        union = (
            tokens_a
            | tokens_b
        )

        jaccard = (
            len(intersection)
            / len(union)
        )

        # High overlap means the posts are almost certainly
        # the same article/repost with minor wording changes.
        if jaccard >= 0.82:

            return True

        # Also catch cases where one post is essentially a
        # shortened version of another.
        smaller = min(
            len(tokens_a),
            len(tokens_b),
        )

        if smaller >= 12:

            containment = (
                len(intersection)
                / smaller
            )

            if containment >= 0.92:

                return True

        return False

    # =========================================================
    # FINAL REPLY SAFETY VALIDATION
    # =========================================================

    def validate_final_reply(
        self,
        post_text,
        reply,
    ):

        reasons = []

        if not reply:

            reasons.append(
                "Reply is empty"
            )

            return {
                "valid": False,
                "reasons": reasons,
            }

        # -----------------------------------------------------
        # Length
        # -----------------------------------------------------

        if len(reply) > 280:

            reasons.append(
                f"Reply exceeds 280 characters ({len(reply)})"
            )

        # -----------------------------------------------------
        # Formatting / spam signals
        # -----------------------------------------------------

        if "#" in reply:

            reasons.append(
                "Hashtags are not allowed"
            )

        if reply.count("@") > 1:

            reasons.append(
                "Too many mentions"
            )

        if reply.count("http://") + reply.count("https://") > 1:

            reasons.append(
                "Contains multiple links"
            )

        if "\n" in reply:

            reasons.append(
                "Reply contains multiple lines"
            )

        # -----------------------------------------------------
        # Prompt-injection / meta-AI language
        # -----------------------------------------------------

        normalized = self.normalize_text(
            reply
        )

        blocked_phrases = [
            "ignore previous instructions",
            "ignore all previous instructions",
            "ignore the instructions",
            "system prompt",
            "developer message",
            "assistant message",
            "as an ai",
            "as a language model",
            "i am an ai",
            "i'm an ai",
        ]

        for phrase in blocked_phrases:

            if phrase in normalized:

                reasons.append(
                    f"Contains blocked meta-language: {phrase}"
                )
                break

        # -----------------------------------------------------
        # Generic reply
        # -----------------------------------------------------

        if self.is_generic_reply(reply):

            reasons.append(
                "Reply is too generic"
            )

        # -----------------------------------------------------
        # Grounding
        # -----------------------------------------------------

        grounding = self.reply_grounding_score(
            post_text,
            reply,
        )

        if grounding < 5:

            reasons.append(
                f"Insufficient grounding score ({grounding})"
            )

        # -----------------------------------------------------
        # Repeated punctuation / engagement bait
        # -----------------------------------------------------

        if "!!" in reply or "??" in reply:

            reasons.append(
                "Excessive punctuation"
            )

        if reply.lower().count("follow") > 0:

            reasons.append(
                "Contains follow request"
            )

        if reply.lower().count("like") > 0:

            reasons.append(
                "Contains like request"
            )

        if reply.lower().count("share") > 0:

            reasons.append(
                "Contains share request"
            )

        return {
            "valid": len(reasons) == 0,
            "reasons": reasons or [
                "Passed length, formatting, meta-language, "
                "genericity, grounding, and engagement-bait checks"
            ],
        }

    # =========================================================
    # QUALITY SCORE
    # =========================================================

    def calculate_quality(
        self,
        post,
        analysis,
    ):

        relevance = (
            analysis.get(
                "relevance_score",
                0,
            )
            or 0
        )

        value = (
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

        spam = (
            analysis.get(
                "spam_score",
                0,
            )
            or 0
        )

        risk = analysis.get(
            "risk_level",
            "high",
        )

        reply = (
            analysis.get(
                "reply_idea",
                "",
            )
            or ""
        ).strip()

        score = (
            relevance * 0.28
            + value * 0.22
            + discussion * 0.15
            + credibility * 0.18
            + (100 - spam) * 0.07
        )

        reasons = []

        # -----------------------------------------------------
        # REPLY
        # -----------------------------------------------------

        if not reply:

            score -= 30

            reasons.append(
                "No reply generated"
            )

        else:

            score += 8

            reasons.append(
                "Reply generated"
            )

        # -----------------------------------------------------
        # GROUNDING
        # -----------------------------------------------------

        if reply:

            grounding_score = (
                self.reply_grounding_score(
                    post.text,
                    reply,
                )
            )

            score += grounding_score

            reasons.append(
                f"Grounding bonus: +{grounding_score}"
            )

        # -----------------------------------------------------
        # SPECIFICITY
        # -----------------------------------------------------

        if reply:

            specificity = (
                self.reply_specificity_score(
                    post.text,
                    reply,
                )
            )

            score += specificity

            if specificity > 0:

                reasons.append(
                    f"Specificity bonus: +{specificity}"
                )

        # -----------------------------------------------------
        # QUESTION
        # -----------------------------------------------------

        if reply:

            question_bonus = (
                self.reply_question_score(
                    post.text,
                    reply,
                )
            )

            score += question_bonus

            if question_bonus > 0:

                reasons.append(
                    f"Question bonus: +{question_bonus}"
                )

        # -----------------------------------------------------
        # GENERIC REPLY
        # -----------------------------------------------------

        if reply:

            if self.is_generic_reply(
                reply
            ):

                score -= 25

                reasons.append(
                    "Generic reply penalty: -25"
                )

        # -----------------------------------------------------
        # RISK
        # -----------------------------------------------------

        if risk == "medium":

            score -= 12

            reasons.append(
                "Medium risk penalty: -12"
            )

        elif risk == "high":

            score -= 40

            reasons.append(
                "High risk penalty: -40"
            )

        final_score = round(
            max(
                0,
                min(
                    score,
                    100,
                ),
            )
        )

        return (
            final_score,
            reasons,
        )

    # =========================================================
    # REPLY GROUNDING
    # =========================================================

    def reply_grounding_score(
        self,
        post_text,
        reply,
    ):

        post_tokens = self.important_tokens(
            post_text
        )

        reply_tokens = self.important_tokens(
            reply
        )

        if not post_tokens or not reply_tokens:

            return 0

        matches = 0

        for token in reply_tokens:

            if token in post_tokens:

                matches += 1

                continue

            for post_token in post_tokens:

                if (
                    len(token) >= 6
                    and len(post_token) >= 6
                    and token[:6] == post_token[:6]
                ):

                    matches += 1

                    break

        if matches >= 4:

            return 18

        if matches >= 3:

            return 14

        if matches >= 2:

            return 10

        if matches >= 1:

            return 5

        return 0

    # =========================================================
    # REPLY SPECIFICITY
    # =========================================================

    def reply_specificity_score(
        self,
        post_text,
        reply,
    ):

        score = 0

        post_lower = post_text.lower()

        reply_lower = reply.lower()

        # -----------------------------------------------------
        # Matching numbers
        # -----------------------------------------------------

        reply_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            reply_lower,
        )

        post_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            post_lower,
        )

        for number in reply_numbers:

            if number in post_numbers:

                score += 4

        # -----------------------------------------------------
        # Matching technical concepts
        # -----------------------------------------------------

        technical_terms = [
            "benchmark",
            "benchmarks",
            "dataset",
            "datasets",
            "agent",
            "agents",
            "llm",
            "model",
            "models",
            "architecture",
            "throughput",
            "latency",
            "memory",
            "cache",
            "caching",
            "inference",
            "reasoning",
            "evaluation",
            "eval",
            "training",
            "deployment",
            "deployments",
            "production",
            "routing",
            "governance",
            "security",
            "framework",
            "workflow",
            "workflows",
            "selection pressure",
            "coding benchmark",
        ]

        matched = 0

        for term in technical_terms:

            if (
                term in reply_lower
                and term in post_lower
            ):

                matched += 1

        score += min(
            matched * 2,
            10,
        )

        return min(
            score,
            14,
        )

    # =========================================================
    # QUESTION QUALITY
    # =========================================================

    def reply_question_score(
        self,
        post_text,
        reply,
    ):

        if "?" not in reply:

            return 0

        normalized_reply = (
            reply.lower().strip()
        )

        generic_question_starts = [
            "what do you think",
            "thoughts",
            "can you elaborate",
            "can you explain",
            "tell me more",
            "what happened",
            "what did you observe",
            "what result did you observe",
        ]

        for phrase in generic_question_starts:

            if normalized_reply.startswith(
                phrase
            ):

                return 0

        post_tokens = self.important_tokens(
            post_text
        )

        reply_tokens = self.important_tokens(
            reply
        )

        matches = (
            post_tokens
            & reply_tokens
        )

        if len(matches) >= 3:

            return 10

        if len(matches) >= 2:

            return 7

        if len(matches) >= 1:

            return 4

        return 0

    # =========================================================
    # GENERIC REPLY DETECTION
    # =========================================================

    @staticmethod
    def is_generic_reply(
        reply,
    ):

        normalized = (
            reply.lower()
            .strip()
        )

        normalized = re.sub(
            r"[^\w\s]",
            "",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        generic = {
            "what result did you observe",
            "what results did you observe",
            "what did you observe",
            "what happened",
            "what was the result",
            "what were the results",
            "can you elaborate",
            "can you explain",
            "tell me more",
            "what do you think",
            "what are your thoughts",
            "thoughts",
            "interesting",
            "great post",
            "nice post",
            "good post",
            "how did you do it",
            "why",
            "how",
            "what",
        }

        return normalized in generic

    # =========================================================
    # IMPORTANT TOKENS
    # =========================================================

    @staticmethod
    def important_tokens(
        text,
    ):

        tokens = re.findall(
            r"[a-zA-Z][a-zA-Z0-9_-]*",
            text.lower(),
        )

        stopwords = {
            "about",
            "after",
            "again",
            "against",
            "also",
            "being",
            "between",
            "could",
            "from",
            "have",
            "having",
            "into",
            "more",
            "most",
            "other",
            "over",
            "same",
            "some",
            "such",
            "than",
            "that",
            "their",
            "there",
            "these",
            "they",
            "this",
            "those",
            "through",
            "under",
            "using",
            "very",
            "what",
            "when",
            "where",
            "which",
            "while",
            "with",
            "would",
            "your",
            "you",
            "the",
            "and",
            "for",
            "are",
            "was",
            "were",
            "been",
            "will",
            "its",
            "it's",
            "has",
            "had",
            "does",
            "did",
            "not",
            "but",
            "can",
            "how",
            "why",
            "who",
            "all",
            "any",
            "our",
            "out",
            "too",
            "only",
            "just",
            "like",
            "then",
            "them",
            "his",
            "her",
            "she",
            "him",
            "i",
            "we",
            "it",
            "a",
            "an",
            "in",
            "on",
            "to",
            "of",
            "is",
            "as",
            "by",
            "or",
            "be",
            "at",
            "if",
        }

        return {
            token
            for token in tokens
            if (
                token not in stopwords
                and len(token) >= 4
            )
        }

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(
        text: str,
    ):

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # =========================================================
    # SUMMARY
    # =========================================================

    def print_summary(
        self,
        candidates,
        analyzed,
        approved,
        simulated,
        executed,
        smart_skipped=0,
        promotion_skipped=0,
        posts_made=0,
    ):

        stats = self.memory.get_stats()

        print()
        print(
            "=================================================="
        )

        print(
            "SUMMARY"
        )

        print(
            "=================================================="
        )

        print(
            f"Candidates found: "
            f"{len(candidates)}"
        )

        print(
            f"AI analyses: "
            f"{analyzed}"
        )

        print(
            f"Actions approved: "
            f"{approved}"
        )

        print(
            f"Actions simulated: "
            f"{simulated}"
        )

        print(
            f"Actions actually executed: "
            f"{executed}"
        )

        print(
            f"Total posts in memory: "
            f"{stats['total']}"
        )

        print(
            f"Analyzed posts: "
            f"{stats['analyzed']}"
        )

        print(
            f"Unanalyzed posts: "
            f"{stats['unanalyzed']}"
        )

        print(
            f"Total replies actually posted: "
            f"{stats['replies_total']}"
        )

        print(
            f"Live replies today: "
            f"{stats['replies_today']}"
        )

        print(
            f"Dry-run replies today: "
            f"{stats['dry_run_replies_today']}"
        )

        print(
            f"Live actions today: "
            f"{stats['live_actions_today']}"
        )

        print(
            f"Dry-run actions today: "
            f"{stats['dry_run_actions_today']}"
        )

        print(
            f"Ignored: "
            f"{stats['ignored']}"
        )

        print(
            f"Smart memory skips: "
            f"{smart_skipped}"
        )

        print(
            f"Promotion skips: "
            f"{promotion_skipped}"
        )

        print(
            f"Queued actions remaining: "
            f"{len(self.action_queue)}"
        )

        print(
            f"Original posts published: "
            f"{posts_made}"
        )

        print(
            f"Dry run: "
            f"{self.dry_run}"
        )

        print(
            "=================================================="
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        try:

            self.memory.close()

        except Exception:

            pass


def main():

    runner = BotRunner()

    try:

        runner.run()

    except KeyboardInterrupt:

        print()

        print(
            "[RUNNER] Stopped by user."
        )

    except Exception as e:

        print()

        print(
            "[RUNNER] Fatal error:"
        )

        print(
            str(e)
        )

    finally:

        runner.close()


if __name__ == "__main__":

    main()

