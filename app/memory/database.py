import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class BotMemory:

    def __init__(
        self,
        db_path: str = "data/bot.db",
    ):

        self.db_path = Path(
            db_path
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self.create_tables()

    # =========================================================
    # TABLES
    # =========================================================

    def create_tables(self):

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uri TEXT UNIQUE NOT NULL,
                cid TEXT,
                author_handle TEXT,
                author_display_name TEXT,
                text TEXT,
                created_at TEXT,
                likes INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                reposts INTEGER DEFAULT 0,
                discovered_at TEXT NOT NULL,

                relevance_score INTEGER DEFAULT 0,
                content_value_score INTEGER DEFAULT 0,
                discussion_score INTEGER DEFAULT 0,
                credibility_score INTEGER DEFAULT 0,
                spam_score INTEGER DEFAULT 0,
                opportunity_score INTEGER DEFAULT 0,

                risk_level TEXT,
                action TEXT,
                reply_style TEXT,
                reason TEXT,
                reply_text TEXT,

                analyzed INTEGER DEFAULT 0,
                executed INTEGER DEFAULT 0
            )
            """
        )

        # -----------------------------------------------------
        # Activity table
        #
        # This records actual/simulated actions separately from
        # post analysis.
        # -----------------------------------------------------

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                uri TEXT,
                action TEXT NOT NULL,

                mode TEXT NOT NULL,

                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    # =========================================================
    # POST MEMORY
    # =========================================================

    def has_seen(
        self,
        uri: str,
    ) -> bool:

        result = self.connection.execute(
            """
            SELECT 1
            FROM posts
            WHERE uri = ?
            LIMIT 1
            """,
            (uri,),
        ).fetchone()

        return result is not None

    def has_analyzed(
        self,
        uri: str,
    ) -> bool:

        result = self.connection.execute(
            """
            SELECT 1
            FROM posts
            WHERE uri = ?
            AND analyzed = 1
            LIMIT 1
            """,
            (uri,),
        ).fetchone()

        return result is not None

    def save_discovered_post(
        self,
        post,
    ):

        self.connection.execute(
            """
            INSERT OR IGNORE INTO posts (
                uri,
                cid,
                author_handle,
                author_display_name,
                text,
                created_at,
                likes,
                replies,
                reposts,
                discovered_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                post.uri,
                post.cid,
                post.author_handle,
                post.author_display_name,
                post.text,
                post.created_at.isoformat(),
                post.likes,
                post.replies,
                post.reposts,
                self.utc_now(),
            ),
        )

        self.connection.commit()

    def save_post(
        self,
        post,
    ):

        self.save_discovered_post(
            post
        )

    # =========================================================
    # ANALYSIS MEMORY
    # =========================================================

    def save_analysis(
        self,
        uri: str,
        analysis: dict,
    ):

        opportunity_score = (
            analysis.get(
                "opportunity_score"
            )
        )

        if opportunity_score is None:

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

            opportunity_score = (
                content_value * 0.6
                + discussion * 0.4
            )

        self.connection.execute(
            """
            UPDATE posts
            SET
                relevance_score = ?,
                content_value_score = ?,
                discussion_score = ?,
                credibility_score = ?,
                spam_score = ?,
                opportunity_score = ?,
                risk_level = ?,
                action = ?,
                reply_style = ?,
                reason = ?,
                reply_text = ?,
                analyzed = 1
            WHERE uri = ?
            """,
            (
                analysis.get(
                    "relevance_score",
                    0,
                ),

                analysis.get(
                    "content_value_score",
                    0,
                ),

                analysis.get(
                    "discussion_score",
                    0,
                ),

                analysis.get(
                    "credibility_score",
                    0,
                ),

                analysis.get(
                    "spam_score",
                    0,
                ),

                opportunity_score,

                analysis.get(
                    "risk_level",
                    "",
                ),

                analysis.get(
                    "action",
                    "ignore",
                ),

                analysis.get(
                    "reply_style",
                    "",
                ),

                analysis.get(
                    "reason",
                    "",
                ),

                analysis.get(
                    "reply_idea",
                    "",
                ),

                uri,
            ),
        )

        self.connection.commit()

    # =========================================================
    # ACTIVITY
    # =========================================================

    def record_activity(
        self,
        uri: str,
        action: str,
        mode: str,
    ):
        """
        Record an action attempt.

        mode:
            dry_run
            live

        action:
            reply
            post
            quote
            follow
            unfollow
        """

        if not action:
            return

        if not mode:
            mode = "unknown"

        self.connection.execute(
            """
            INSERT INTO activity (
                uri,
                action,
                mode,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                uri,
                action,
                mode,
                self.utc_now(),
            ),
        )

        self.connection.commit()

    def mark_executed(
        self,
        uri: str,
    ):

        self.connection.execute(
            """
            UPDATE posts
            SET executed = 1
            WHERE uri = ?
            """,
            (uri,),
        )

        self.connection.commit()

    # =========================================================
    # DAILY ACTIVITY
    # =========================================================

    def get_activity_today(
        self,
        mode: str = None,
        action: str = None,
    ) -> int:

        query = """
            SELECT COUNT(*)
            FROM activity
            WHERE date(created_at) = date('now')
        """

        params = []

        if mode is not None:

            query += """
                AND mode = ?
            """

            params.append(
                mode
            )

        if action is not None:

            query += """
                AND action = ?
            """

            params.append(
                action
            )

        result = self.connection.execute(
            query,
            params,
        ).fetchone()

        return int(
            result[0]
        )

    def get_live_replies_today(
        self,
    ) -> int:

        return self.get_activity_today(
            mode="live",
            action="reply",
        )

    def get_dry_run_replies_today(
        self,
    ) -> int:

        return self.get_activity_today(
            mode="dry_run",
            action="reply",
        )

    def get_live_actions_today(
        self,
    ) -> int:

        return self.get_activity_today(
            mode="live"
        )

    def get_dry_run_actions_today(
        self,
    ) -> int:

        return self.get_activity_today(
            mode="dry_run"
        )

    # =========================================================
    # STATS
    # =========================================================

    def get_stats(
        self,
    ) -> dict:

        total = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM posts
            """
        ).fetchone()[0]

        analyzed = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM posts
            WHERE analyzed = 1
            """
        ).fetchone()[0]

        unanalyzed = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM posts
            WHERE analyzed = 0
            """
        ).fetchone()[0]

        ignored = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM posts
            WHERE action = 'ignore'
            """
        ).fetchone()[0]

        replies_total = (
            self.get_activity_count(
                mode="live",
                action="reply",
            )
        )

        replies_today = (
            self.get_live_replies_today()
        )

        dry_run_replies_today = (
            self.get_dry_run_replies_today()
        )

        live_actions_today = (
            self.get_live_actions_today()
        )

        dry_run_actions_today = (
            self.get_dry_run_actions_today()
        )

        return {
            "total": total,

            "analyzed": analyzed,

            "unanalyzed": unanalyzed,

            "replies": replies_today,

            "replies_today": replies_today,

            "replies_total": replies_total,

            "ignored": ignored,

            "dry_run_replies_today": (
                dry_run_replies_today
            ),

            "live_actions_today": (
                live_actions_today
            ),

            "dry_run_actions_today": (
                dry_run_actions_today
            ),
        }

    def get_activity_count(
        self,
        mode: str = None,
        action: str = None,
    ) -> int:

        query = """
            SELECT COUNT(*)
            FROM activity
            WHERE 1 = 1
        """

        params = []

        if mode is not None:

            query += """
                AND mode = ?
            """

            params.append(
                mode
            )

        if action is not None:

            query += """
                AND action = ?
            """

            params.append(
                action
            )

        result = self.connection.execute(
            query,
            params,
        ).fetchone()

        return int(
            result[0]
        )

    # =========================================================
    # DATABASE HELPERS
    # =========================================================

    @staticmethod
    def utc_now():

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.connection.close()
