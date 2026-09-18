import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from atproto import Client, models
from atproto_client.models.utils import create_strong_ref
from dotenv import load_dotenv

HTTP_TIMEOUT = httpx.Timeout(connect=15, read=60, write=30, pool=15)
LOGIN_RETRIES = 3
LOGIN_RETRY_DELAY = 3

load_dotenv()


@dataclass
class Post:
    uri: str
    cid: str
    text: str
    author_handle: str
    author_display_name: str
    created_at: datetime
    likes: int = 0
    replies: int = 0
    reposts: int = 0
    url: Optional[str] = None


class BlueskyClient:
    def __init__(self):
        handle = os.getenv("BLUESKY_HANDLE")
        app_password = os.getenv("BLUESKY_APP_PASSWORD")

        if not handle:
            raise ValueError("BLUESKY_HANDLE is missing from .env")
        if not app_password:
            raise ValueError("BLUESKY_APP_PASSWORD is missing from .env")

        self.client = Client()

        # NOTE: trust_env=False + explicit proxy. Some sandboxed environments
        # export no_proxy entries (e.g. bracketed IPv6 like "[::1]") that this
        # httpx version fails to parse ("Invalid port: ':1]'"). Passing the
        # proxy explicitly avoids that while still working behind a proxy.
        # On a normal machine with no proxy env vars, this just connects directly.
        proxy_url = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")

        old_http_client = getattr(self.client.request, "_client", None)
        self.client.request._client = httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            trust_env=False,
            proxy=proxy_url,
            # trust_env=False also skips SSL_CERT_FILE, so pass the CA bundle
            # explicitly. Falls back to default verification on normal machines.
            verify=os.getenv("SSL_CERT_FILE") or True,
        )

        if old_http_client is not None:
            try:
                old_http_client.close()
            except Exception:
                pass

        self._login_with_retry(handle, app_password)

    def _login_with_retry(self, handle: str, app_password: str) -> None:
        last_error = None

        for attempt in range(1, LOGIN_RETRIES + 1):
            try:
                print(f"[BLUESKY] Logging in (attempt {attempt}/{LOGIN_RETRIES})...")
                # fetch_bsky_profile=False: the SDK's post-login profile lookup
                # (app.bsky.actor.getProfile) can hang in sandboxed/proxied
                # networks even when createSession succeeds. The session is
                # what the bot needs, so skip the lookup.
                self.client.login(handle, app_password, fetch_bsky_profile=False)
                print("[BLUESKY] Login successful.")
                print(f"Logged into Bluesky as {handle}")
                self._ensure_profile(handle)
                return
            except Exception as exc:
                last_error = exc
                if attempt < LOGIN_RETRIES:
                    print(f"[BLUESKY] Login failed: {exc}. Retrying in {LOGIN_RETRY_DELAY}s...")
                    time.sleep(LOGIN_RETRY_DELAY)

        raise RuntimeError(
            f"Bluesky login failed after {LOGIN_RETRIES} attempts: {last_error}"
        )

    def _ensure_profile(self, handle: str) -> None:
        """Populate ``client.me`` after a session-only login.

        ``login(fetch_bsky_profile=False)`` leaves ``client.me`` as None, but
        the SDK's ``send_post`` derives the repo from ``me.did`` and raises
        ``LoginRequiredError`` ("you must be logged in") without it.

        First try the profile lookup directly (it works now that the HTTP
        client is configured explicitly); if that fails, fall back to a
        minimal profile built from the session's DID/handle, which is all
        publishing needs.
        """
        try:
            profile = self.client.app.bsky.actor.get_profile(
                models.AppBskyActorGetProfile.Params(actor=handle)
            )
            self.client.me = profile
            print("[BLUESKY] Profile loaded.")
            return
        except Exception as exc:
            print(f"[BLUESKY] Profile lookup failed ({exc}); using session identity.")

        session = getattr(self.client, "_session", None)
        did = getattr(session, "did", None) if session else None
        session_handle = getattr(session, "handle", None) if session else None
        if not did:
            raise RuntimeError(
                "Bluesky login succeeded but no session DID is available."
            )

        self.client.me = models.AppBskyActorDefs.ProfileViewDetailed(
            did=did,
            handle=session_handle or handle,
        )
        print("[BLUESKY] Publishing identity ready.")

    def search_posts(
        self,
        query: str,
        limit: int = 20,
        sort: str = "latest",
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> list[Post]:

        params = {
            "q": query,
            "limit": min(max(int(limit), 1), 100),
            "sort": sort,
        }

        if since:
            params["since"] = since
        if until:
            params["until"] = until

        response = self.client.app.bsky.feed.search_posts(params=params)

        posts = []

        for item in response.posts:
            record = item.record

            created_at = datetime.fromisoformat(
                record.created_at.replace("Z", "+00:00")
            )

            author_handle = item.author.handle
            post_url = (
                f"https://bsky.app/profile/{author_handle}/post/"
                f"{item.uri.split('/')[-1]}"
            )

            posts.append(
                Post(
                    uri=item.uri,
                    cid=item.cid,
                    text=record.text,
                    author_handle=author_handle,
                    author_display_name=item.author.display_name or "",
                    created_at=created_at,
                    likes=item.like_count or 0,
                    replies=item.reply_count or 0,
                    reposts=item.repost_count or 0,
                    url=post_url,
                )
            )

        return posts

    def search_posts_window(
        self,
        query: str,
        limit: int = 25,
        hours: int = 168,
    ) -> list[Post]:
        """
        Search established posts over a wider window plus recent posts.

        1. TOP: up to 7 days old, ranked by search popularity.
        2. LATEST: last 24 hours, so fresh posts are still discovered.
        """

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=max(1, int(hours)))
        recent_start = now - timedelta(hours=min(24, max(1, int(hours))))

        top_posts = self.search_posts(
            query=query,
            limit=limit,
            sort="top",
            since=window_start.isoformat().replace("+00:00", "Z"),
            until=now.isoformat().replace("+00:00", "Z"),
        )

        latest_posts = self.search_posts(
            query=query,
            limit=limit,
            sort="latest",
            since=recent_start.isoformat().replace("+00:00", "Z"),
            until=now.isoformat().replace("+00:00", "Z"),
        )

        merged = {}

        for post in top_posts + latest_posts:
            existing = merged.get(post.uri)

            if existing is None:
                merged[post.uri] = post
                continue

            old_engagement = (
                existing.likes
                + existing.replies * 3
                + existing.reposts * 4
            )
            new_engagement = (
                post.likes
                + post.replies * 3
                + post.reposts * 4
            )

            if new_engagement > old_engagement:
                merged[post.uri] = post

        return list(merged.values())

    def send_post(self, text: str):
        return self.client.send_post(text)

    def reply_to_post(self, text: str, post: Post):
        # create_strong_ref is a utility function, not an attribute of
        # the generated `models` package. Calling `models.create_strong_ref`
        # makes Python try to call a module, which causes:
        #   TypeError: 'module' object is not callable
        #
        # `post` already has the required `uri` and `cid` fields, so it can
        # be passed directly to the helper.
        parent = create_strong_ref(post)

        reply_ref = models.AppBskyFeedPost.ReplyRef(
            root=parent,
            parent=parent,
        )

        return self.client.send_post(
            text=text,
            reply_to=reply_ref,
        )

