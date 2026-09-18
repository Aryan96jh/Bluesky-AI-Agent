import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


class SmartMemory:
    """
    Persistent memory layer for author, topic, and reply history.

    Memory is intentionally separate from the existing SQLite
    BotMemory so historical database data remains untouched.

    Modes:
        dry_run
            Used for testing. Dry-run interactions can participate
            in duplicate topic/reply detection, but they DO NOT
            create a live author cooldown.

        live
            Represents a real interaction with Bluesky. Live
            interactions create the real author cooldown.
    """

    def __init__(self, path=None):

        self.path = Path(
            path
            or os.getenv(
                "SMART_MEMORY_FILE",
                "data/smart_memory.json",
            )
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data = {
            "authors": {},
            "interactions": [],
        }

        self._load()

    # =========================================================
    # LOAD
    # =========================================================

    def _load(self):

        if not self.path.exists():
            return

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as f:

                loaded = json.load(f)

            if not isinstance(
                loaded,
                dict,
            ):
                return

            authors = loaded.get(
                "authors",
                {},
            )

            interactions = loaded.get(
                "interactions",
                [],
            )

            if isinstance(
                authors,
                dict,
            ):
                self.data["authors"] = authors

            if isinstance(
                interactions,
                list,
            ):
                self.data["interactions"] = interactions

        except (
            OSError,
            json.JSONDecodeError,
        ):

            # Smart memory is auxiliary.
            # Never prevent the bot from starting.

            self.data = {
                "authors": {},
                "interactions": [],
            }

    # =========================================================
    # SAVE
    # =========================================================

    def _save(self):

        tmp = self.path.with_suffix(
            ".tmp"
        )

        try:

            with tmp.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    self.data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            tmp.replace(
                self.path
            )

        except OSError:

            # Auxiliary memory should never
            # crash the bot.

            try:

                if tmp.exists():
                    tmp.unlink()

            except OSError:
                pass

    # =========================================================
    # TIME
    # =========================================================

    @staticmethod
    def _now():

        return datetime.now(
            timezone.utc
        )

    @staticmethod
    def _parse_time(value):

        # Important:
        # New authors have None for previous interaction.

        if not value:
            return None

        if isinstance(
            value,
            datetime,
        ):

            parsed = value

        else:

            try:

                parsed = datetime.fromisoformat(
                    str(value).replace(
                        "Z",
                        "+00:00",
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                return None

        # Normalize old naive timestamps.

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokens(text):

        words = re.findall(
            r"[a-zA-Z][a-zA-Z0-9_-]*",
            (text or "").lower(),
        )

        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "they",
            "their",
            "there",
            "about",
            "would",
            "could",
            "should",
            "have",
            "has",
            "been",
            "were",
            "what",
            "when",
            "where",
            "which",
            "while",
            "into",
            "than",
            "then",
            "them",
            "your",
            "you",
            "for",
            "and",
            "the",
            "are",
            "was",
            "will",
            "how",
            "why",
            "who",
            "does",
            "did",
            "its",
            "it's",
            "also",
            "just",
            "more",
            "some",
            "very",
            "only",
            "using",
            "used",
            "use",
            "can",
            "not",
            "but",
            "our",
            "out",
        }

        return {
            word
            for word in words
            if len(word) >= 4
            and word not in stopwords
        }

    # =========================================================
    # PRUNE OLD MEMORY
    # =========================================================

    def _prune(
        self,
        days=60,
    ):

        cutoff = (
            self._now()
            - timedelta(
                days=days
            )
        )

        interactions = self.data.get(
            "interactions",
            [],
        )

        kept = []

        for item in interactions:

            if not isinstance(
                item,
                dict,
            ):
                continue

            timestamp = self._parse_time(
                item.get(
                    "timestamp"
                )
            )

            # Keep malformed entries rather than
            # silently deleting potentially useful data.

            if timestamp is None:

                kept.append(
                    item
                )

                continue

            if timestamp >= cutoff:

                kept.append(
                    item
                )

        self.data["interactions"] = kept

    # =========================================================
    # AUTHOR STATUS
    # =========================================================

    def author_status(
        self,
        handle,
        cooldown_hours=72,
    ):

        if not handle:

            return {
                "skip": False,
                "interactions": 0,
                "live_interactions": 0,
                "dry_run_interactions": 0,
                "hours_since": None,
            }

        record = (
            self.data
            .get(
                "authors",
                {},
            )
            .get(
                handle,
                {},
            )
        )

        if not isinstance(
            record,
            dict,
        ):

            record = {}

        interactions = int(
            record.get(
                "interactions",
                0,
            )
            or 0
        )

        live_interactions = int(
            record.get(
                "live_interactions",
                0,
            )
            or 0
        )

        dry_run_interactions = int(
            record.get(
                "dry_run_interactions",
                0,
            )
            or 0
        )

        # -----------------------------------------------------
        # IMPORTANT
        #
        # Only a REAL live interaction creates the live
        # author cooldown.
        #
        # This prevents our DRY_RUN testing from poisoning
        # the live author history.
        # -----------------------------------------------------

        last = self._parse_time(
            record.get(
                "last_live_interaction"
            )
        )

        # Backwards compatibility:
        #
        # Old smart_memory files may have only
        # "last_interaction".
        #
        # We deliberately DO NOT use that old field as a
        # live cooldown because old records may have been
        # generated during dry-run testing.

        if last is None:

            return {
                "skip": False,
                "interactions": interactions,
                "live_interactions": live_interactions,
                "dry_run_interactions": dry_run_interactions,
                "hours_since": None,
            }

        hours_since = max(
            (
                self._now()
                - last
            ).total_seconds()
            / 3600,
            0,
        )

        return {
            "skip": (
                hours_since
                < cooldown_hours
            ),
            "interactions": interactions,
            "live_interactions": live_interactions,
            "dry_run_interactions": dry_run_interactions,
            "hours_since": hours_since,
        }

    # =========================================================
    # AUTHOR COOLDOWN
    # =========================================================

    def should_skip_author(
        self,
        handle,
        cooldown_hours=72,
    ):

        status = self.author_status(
            handle=handle,
            cooldown_hours=cooldown_hours,
        )

        return status["skip"]

    # =========================================================
    # TOPIC OVERLAP
    # =========================================================

    def topic_overlap(
        self,
        text,
        days=14,
        threshold=0.35,
    ):

        current = self._tokens(
            text
        )

        if len(current) < 3:
            return None

        cutoff = (
            self._now()
            - timedelta(
                days=days
            )
        )

        best = None

        for item in self.data.get(
            "interactions",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            timestamp = self._parse_time(
                item.get(
                    "timestamp"
                )
            )

            if timestamp is None:
                continue

            if timestamp < cutoff:
                continue

            previous = set(
                item.get(
                    "topic_tokens",
                    [],
                )
            )

            if len(previous) < 3:
                continue

            intersection = len(
                current & previous
            )

            union = len(
                current | previous
            )

            similarity = (
                intersection / union
                if union
                else 0
            )

            if (
                similarity >= threshold
                and (
                    best is None
                    or similarity
                    > best["similarity"]
                )
            ):

                best = {
                    "similarity": similarity,
                    "author": item.get(
                        "author",
                        "",
                    ),
                    "reply": item.get(
                        "reply",
                        "",
                    ),
                    "timestamp": item.get(
                        "timestamp",
                        "",
                    ),
                    "mode": item.get(
                        "mode",
                        "",
                    ),
                }

        return best

    # =========================================================
    # SHOULD SKIP TOPIC
    # =========================================================

    def should_skip_topic(
        self,
        text,
        days=14,
        threshold=0.35,
    ):

        match = self.topic_overlap(
            text=text,
            days=days,
            threshold=threshold,
        )

        return match is not None

    # =========================================================
    # SIMILAR REPLY
    # =========================================================

    def similar_reply(
        self,
        reply,
        days=30,
        threshold=0.60,
    ):

        current = self._tokens(
            reply
        )

        if len(current) < 2:
            return None

        cutoff = (
            self._now()
            - timedelta(
                days=days
            )
        )

        best = None

        for item in self.data.get(
            "interactions",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            timestamp = self._parse_time(
                item.get(
                    "timestamp"
                )
            )

            if timestamp is None:
                continue

            if timestamp < cutoff:
                continue

            previous = set(
                item.get(
                    "reply_tokens",
                    [],
                )
            )

            if len(previous) < 2:
                continue

            intersection = len(
                current & previous
            )

            union = len(
                current | previous
            )

            similarity = (
                intersection / union
                if union
                else 0
            )

            if (
                similarity >= threshold
                and (
                    best is None
                    or similarity
                    > best["similarity"]
                )
            ):

                best = {
                    "similarity": similarity,
                    "author": item.get(
                        "author",
                        "",
                    ),
                    "reply": item.get(
                        "reply",
                        "",
                    ),
                    "timestamp": item.get(
                        "timestamp",
                        "",
                    ),
                    "mode": item.get(
                        "mode",
                        "",
                    ),
                }

        return best

    # =========================================================
    # SHOULD SKIP REPLY
    # =========================================================

    def should_skip_reply(
        self,
        reply,
        days=30,
        threshold=0.60,
    ):

        match = self.similar_reply(
            reply=reply,
            days=days,
            threshold=threshold,
        )

        return match is not None

    # =========================================================
    # RECORD INTERACTION
    # =========================================================

    def record_interaction(
        self,
        author,
        post_text,
        reply,
        mode="dry_run",
        action="reply",
    ):

        now = (
            self._now()
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            )
        )

        # Normalize mode.

        mode = (
            str(mode or "dry_run")
            .strip()
            .lower()
        )

        if mode not in {
            "dry_run",
            "live",
        }:

            mode = "dry_run"

        authors = self.data.setdefault(
            "authors",
            {},
        )

        record = authors.setdefault(
            author,
            {
                "seen": 0,
                "interactions": 0,
                "live_interactions": 0,
                "dry_run_interactions": 0,
                "last_seen": None,
                "last_interaction": None,
                "last_live_interaction": None,
                "last_dry_run_interaction": None,
            },
        )

        if not isinstance(
            record,
            dict,
        ):

            record = {
                "seen": 0,
                "interactions": 0,
                "live_interactions": 0,
                "dry_run_interactions": 0,
                "last_seen": None,
                "last_interaction": None,
                "last_live_interaction": None,
                "last_dry_run_interaction": None,
            }

            authors[author] = record

        # -----------------------------------------------------
        # Backwards-compatible defaults
        # -----------------------------------------------------

        record.setdefault(
            "seen",
            0,
        )

        record.setdefault(
            "interactions",
            0,
        )

        record.setdefault(
            "live_interactions",
            0,
        )

        record.setdefault(
            "dry_run_interactions",
            0,
        )

        record.setdefault(
            "last_seen",
            None,
        )

        record.setdefault(
            "last_interaction",
            None,
        )

        record.setdefault(
            "last_live_interaction",
            None,
        )

        record.setdefault(
            "last_dry_run_interaction",
            None,
        )

        # -----------------------------------------------------
        # Seen count
        # -----------------------------------------------------

        record["seen"] = (
            int(
                record.get(
                    "seen",
                    0,
                )
                or 0
            )
            + 1
        )

        record["last_seen"] = now

        # -----------------------------------------------------
        # Interaction counts
        # -----------------------------------------------------

        if action == "reply":

            record["interactions"] = (
                int(
                    record.get(
                        "interactions",
                        0,
                    )
                    or 0
                )
                + 1
            )

            record["last_interaction"] = now

            if mode == "live":

                record["live_interactions"] = (
                    int(
                        record.get(
                            "live_interactions",
                            0,
                        )
                        or 0
                    )
                    + 1
                )

                record[
                    "last_live_interaction"
                ] = now

            else:

                record["dry_run_interactions"] = (
                    int(
                        record.get(
                            "dry_run_interactions",
                            0,
                        )
                        or 0
                    )
                    + 1
                )

                record[
                    "last_dry_run_interaction"
                ] = now

        # -----------------------------------------------------
        # Store interaction
        # -----------------------------------------------------

        topic_tokens = sorted(
            self._tokens(
                post_text
            )
        )

        reply_tokens = sorted(
            self._tokens(
                reply
            )
        )

        interaction = {
            "timestamp": now,
            "author": author or "",
            "action": action or "",
            "mode": mode,
            "post_tokens": topic_tokens,
            "topic_tokens": topic_tokens,
            "reply": (
                reply or ""
            ).strip(),
            "reply_tokens": reply_tokens,
        }

        self.data.setdefault(
            "interactions",
            [],
        ).append(
            interaction
        )

        self._prune()

        self._save()

    # =========================================================
    # MARK AUTHOR SEEN
    # =========================================================

    def mark_seen(
        self,
        author,
    ):

        if not author:
            return

        now = (
            self._now()
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            )
        )

        authors = self.data.setdefault(
            "authors",
            {},
        )

        record = authors.setdefault(
            author,
            {
                "seen": 0,
                "interactions": 0,
                "live_interactions": 0,
                "dry_run_interactions": 0,
                "last_seen": None,
                "last_interaction": None,
                "last_live_interaction": None,
                "last_dry_run_interaction": None,
            },
        )

        if not isinstance(
            record,
            dict,
        ):

            record = {
                "seen": 0,
                "interactions": 0,
                "live_interactions": 0,
                "dry_run_interactions": 0,
                "last_seen": None,
                "last_interaction": None,
                "last_live_interaction": None,
                "last_dry_run_interaction": None,
            }

            authors[author] = record

        record.setdefault(
            "seen",
            0,
        )

        record.setdefault(
            "interactions",
            0,
        )

        record.setdefault(
            "live_interactions",
            0,
        )

        record.setdefault(
            "dry_run_interactions",
            0,
        )

        record.setdefault(
            "last_interaction",
            None,
        )

        record.setdefault(
            "last_live_interaction",
            None,
        )

        record.setdefault(
            "last_dry_run_interaction",
            None,
        )

        record["seen"] = (
            int(
                record.get(
                    "seen",
                    0,
                )
                or 0
            )
            + 1
        )

        record["last_seen"] = now

        self._save()

    # =========================================================
    # AUTHOR INFO
    # =========================================================

    def get_author(
        self,
        handle,
    ):

        record = (
            self.data
            .get(
                "authors",
                {},
            )
            .get(
                handle,
            )
        )

        if not isinstance(
            record,
            dict,
        ):
            return None

        return dict(
            record
        )

    # =========================================================
    # RECENT INTERACTIONS
    # =========================================================

    def recent_interactions(
        self,
        limit=10,
    ):

        interactions = self.data.get(
            "interactions",
            [],
        )

        if not isinstance(
            interactions,
            list,
        ):
            return []

        if limit <= 0:
            return []

        return list(
            reversed(
                interactions[-limit:]
            )
        )

    # =========================================================
    # LIVE REPLIES TODAY
    # =========================================================

    def get_live_replies_today(self):
        """
        Return the number of real live reply interactions
        recorded since the start of the current UTC day.

        Dry-run interactions are deliberately excluded.
        """

        now = self._now()

        start_of_day = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        count = 0

        for item in self.data.get(
            "interactions",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            if (
                str(
                    item.get(
                        "mode",
                        "",
                    )
                ).lower()
                != "live"
            ):
                continue

            if (
                str(
                    item.get(
                        "action",
                        "",
                    )
                ).lower()
                != "reply"
            ):
                continue

            timestamp = self._parse_time(
                item.get(
                    "timestamp"
                )
            )

            if timestamp is None:
                continue

            if timestamp >= start_of_day:
                count += 1

        return count

    # =========================================================
    # MEMORY COUNTS
    # =========================================================

    def memory_counts(self):

        authors = self.data.get(
            "authors",
            {},
        )

        interactions = self.data.get(
            "interactions",
            [],
        )

        if not isinstance(
            authors,
            dict,
        ):
            authors = {}

        if not isinstance(
            interactions,
            list,
        ):
            interactions = []

        live_interactions = 0
        dry_run_interactions = 0

        for item in interactions:

            if not isinstance(
                item,
                dict,
            ):
                continue

            mode = str(
                item.get(
                    "mode",
                    "",
                )
            ).lower()

            if mode == "live":

                live_interactions += 1

            elif mode == "dry_run":

                dry_run_interactions += 1

        return {
            "authors": len(
                authors
            ),
            "interactions": len(
                interactions
            ),
            "live_interactions": live_interactions,
            "dry_run_interactions": dry_run_interactions,
        }

    # =========================================================
    # STATS
    # =========================================================

    def stats(self):

        self._prune()

        return self.memory_counts()

    # =========================================================
    # PRINT STATS
    # =========================================================

    def print_stats(self):

        stats = self.stats()

        print(
            "[SMART MEMORY] "
            f"Authors={stats['authors']} "
            f"Interactions={stats['interactions']} "
            f"Live={stats['live_interactions']} "
            f"DryRun={stats['dry_run_interactions']}"
        )

    # =========================================================
    # CLEAR MEMORY
    # =========================================================

    def clear(self):

        self.data = {
            "authors": {},
            "interactions": [],
        }

        self._save()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):
        """
        Compatibility method.

        SmartMemory does not maintain an open database
        connection, so there is nothing to close.
        """

        pass
