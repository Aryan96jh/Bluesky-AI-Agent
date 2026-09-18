class ActionEngine:
    def __init__(self, bluesky_client):
        self.bluesky = bluesky_client

    def execute(self, action, post=None, text=None, dry_run=True):
        if action == "ignore":
            return {
                "success": True,
                "action": "ignore",
                "message": "Nothing to do.",
            }

        if action == "reply":
            if not post or not text:
                return {
                    "success": False,
                    "action": "reply",
                    "message": "Missing post or reply text.",
                }

            if dry_run:
                return {
                    "success": True,
                    "action": "reply",
                    "message": "Dry run: reply was not published.",
                    "text": text,
                }

            result = self.bluesky.reply_to_post(
                text=text,
                post=post,
            )

            return {
                "success": True,
                "action": "reply",
                "message": "Reply published.",
                "uri": result.uri,
            }

        if action == "post_idea":
            if not text:
                return {
                    "success": False,
                    "action": "post_idea",
                    "message": "Missing post text.",
                }

            if dry_run:
                return {
                    "success": True,
                    "action": "post_idea",
                    "message": "Dry run: post was not published.",
                    "text": text,
                }

            result = self.bluesky.send_post(text)

            return {
                "success": True,
                "action": "post_idea",
                "message": "Post published.",
                "uri": result.uri,
            }

        if action == "quote":
            return {
                "success": False,
                "action": "quote",
                "message": "Quote posting will be implemented separately.",
            }

        return {
            "success": False,
            "action": action,
            "message": "Unknown action.",
        }
