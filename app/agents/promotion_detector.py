import re
from dataclasses import dataclass
from typing import List


@dataclass
class PromotionResult:
    score: int
    is_promotional: bool
    reasons: List[str]


class PromotionDetector:
    """
    Deterministic detector for obvious promotion/self-promotion.

    The detector distinguishes strong commercial/self-promotional signals
    from ordinary mentions of products, companies, APIs, GitHub repos,
    launches, or technical announcements.

    A URL alone is not promotional.
    A product/company mention alone is not promotional.
    """

    DEFAULT_THRESHOLD = 5

    # Strong calls-to-action. These carry the most weight.
    CTA_PATTERNS = [
        (r"\bcreate\s+(?:an?\s+)?api\s+key\b", 3, "API-key CTA"),
        (r"\bsign\s*up\b", 3, "signup CTA"),
        (
            r"\btry\s+(?:our|my|this)\s+(?:tool|app|product|service|platform)\b",
            4,
            "tool/product CTA",
        ),
        (r"\btry\s+(?:it|this)\b", 1, "trial CTA"),
        (r"\bget\s+started\b", 2, "get-started CTA"),
        (
            r"\buse\s+(?:my|our)\s+(?:code|link|referral)\b",
            5,
            "referral CTA",
        ),
        (r"\breferral\b", 3, "referral language"),
        (r"\bdiscount\b", 3, "discount language"),
        (r"\bfree\s+trial\b", 3, "free-trial CTA"),
        (r"\bbuy\s+(?:now|today)\b", 5, "purchase CTA"),
        (r"\bsubscribe\b", 3, "subscription CTA"),
        (r"\bbook\s+(?:a\s+)?demo\b", 5, "demo CTA"),
        (r"\bdownload\s+(?:now|here|it|our)\b", 3, "download CTA"),
        (r"\bjoin\s+(?:us|now|our)\b", 2, "join CTA"),
        (r"\blimited\s+time\b", 3, "urgency language"),
        (r"\bclaim\s+(?:your|the)\b", 2, "claim CTA"),
        (r"\bstart\s+(?:your|using)\b", 1, "start CTA"),
    ]

    # Marketing/pitch language without an explicit CTA.
    MARKETING_PATTERNS = [
        (
            r"\b(?:streamline|supercharge|boost|unlock|accelerate|transform|simplify|automate)\b"
            r".{0,100}\b(?:with|using|via)\b.{0,100}"
            r"\b(?:platform|product|tool|service|software|agents?|api|sdk)\b",
            4,
            "marketing benefit/product pitch",
        ),
        (
            r"\b(?:our|my)\s+(?:platform|product|tool|app|service|solution|startup)\b",
            2,
            "first-person product pitch",
        ),
        (
            r"\b(?:we|i)\s+(?:built|created|made|launched|released|ship(?:ped)?)\b",
            2,
            "first-person product announcement",
        ),
        (
            r"\b(?:now|today)\s+(?:available|live|open|launched)\b",
            1,
            "availability announcement",
        ),
        (
            r"\b(?:powerful|seamless|revolutionary|game[- ]changing|cutting[- ]edge|best[- ]in[- ]class)\s+"
            r"(?:tool|platform|product|solution|service|app|technology)\b",
            3,
            "marketing adjective",
        ),
        (
            r"\b(?:streamline|supercharge|boost|transform|unlock|accelerate)\s+"
            r"(?:your|the)\s+(?:workflow|business|growth|productivity|development|team|operations)\b",
            3,
            "marketing benefit claim",
        ),
        (
            r"\b(?:save|reduce|cut)\s+(?:hours?|costs?|time)\b",
            2,
            "marketing efficiency claim",
        ),
        (
            r"\b(?:trusted|used|loved)\s+by\s+\d+",
            3,
            "social-proof claim",
        ),
        (
            r"\b(?:customers?|teams?|developers?)\s+(?:love|choose|prefer)\s+(?:our|my)\b",
            3,
            "social-proof language",
        ),
        (
            r"\b(?:learn\s+more|read\s+more)\s+(?:about|here)\b",
            2,
            "learn-more CTA",
        ),
        (
            r"\b(?:check\s+out|check\s+it\s+out)\s+(?:our|my|this)\b",
            3,
            "promotion CTA",
        ),
        (
            r"\b(?:available|live)\s+(?:on|at)\s+(?:our|my)\b",
            2,
            "product availability pitch",
        ),
    ]

    URL_PATTERN = re.compile(
        r"https?://\S+|www\.\S+",
        re.I,
    )

    HASHTAG_PATTERN = re.compile(
        r"(?<!\w)#([A-Za-z0-9_]+)"
    )

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = max(1, int(threshold))

    def analyze(self, text: str) -> PromotionResult:
        text = (text or "").strip()

        if not text:
            return PromotionResult(
                score=0,
                is_promotional=False,
                reasons=[],
            )

        lowered = text.lower()

        score = 0
        reasons = []

        # ---------------------------------------------------------
        # Explicit CTA detection
        # ---------------------------------------------------------

        for pattern, points, reason in self.CTA_PATTERNS:
            if re.search(pattern, lowered):
                score += points
                reasons.append(reason)

        # ---------------------------------------------------------
        # Marketing/self-promotion language
        # ---------------------------------------------------------

        for pattern, points, reason in self.MARKETING_PATTERNS:
            if re.search(pattern, lowered):
                score += points
                reasons.append(reason)

        # ---------------------------------------------------------
        # URL + promotional language
        #
        # A URL alone is NOT promotional.
        # ---------------------------------------------------------

        urls = self.URL_PATTERN.findall(text)

        if urls and reasons:
            score += 2
            reasons.append(
                "external URL paired with promotional language"
            )

        # ---------------------------------------------------------
        # Multiple URLs are a mild commercial signal when combined
        # with first-person or marketing language.
        # ---------------------------------------------------------

        if len(urls) >= 2 and reasons:
            score += 1
            reasons.append(
                "multiple external URLs with promotional language"
            )

        # ---------------------------------------------------------
        # Promotional hashtags
        # ---------------------------------------------------------

        hashtags = self.HASHTAG_PATTERN.findall(text)

        # Explicit disclosure tags are unambiguous promotion signals.
        # A single #ad/#paidlink/#affiliate should be enough to skip the post.
        explicit_promo_tags = {
            "ad",
            "advertisement",
            "sponsored",
            "paidlink",
            "paidpromotion",
            "affiliate",
            "affiliatead",
            "partner",
        }
        explicit_promo_hits = sum(
            1
            for tag in hashtags
            if tag.lower() in explicit_promo_tags
        )
        if explicit_promo_hits:
            score = max(score, 10)
            reasons.append("explicit advertising/affiliate disclosure")

        # Common affiliate URL markers are also strong evidence.
        affiliate_url_patterns = [
            r"[?&](?:tag|aff|affiliate|ref|referral)=",
            r"\b(?:affiliate|referral)\b",
        ]
        if any(re.search(pattern, lowered) for pattern in affiliate_url_patterns):
            score = max(score, 8)
            reasons.append("affiliate/referral link signal")

        promotional_hashtags = {
            "launch",
            "product",
            "startup",
            "saas",
            "discount",
            "giveaway",
            "sale",
            "promo",
            "promotion",
            "free",
            "offer",
            "buildinpublic",
        }

        promo_tag_count = sum(
            1
            for tag in hashtags
            if tag.lower() in promotional_hashtags
        )

        if promo_tag_count >= 2:
            score += 2
            reasons.append(
                "multiple promotional hashtags"
            )
        elif promo_tag_count == 1:
            score += 1
            reasons.append(
                "promotional hashtag"
            )

        # ---------------------------------------------------------
        # First-person launch + link/CTA
        # ---------------------------------------------------------

        first_person_launch = re.search(
            r"\b(?:we|i)\s+"
            r"(?:just\s+)?"
            r"(?:launched|built|released|made|created|shipped)\b",
            lowered,
        )

        launch_cta = re.search(
            r"\b(?:try|sign\s*up|get\s+started|check\s+out|learn\s+more)\b",
            lowered,
        )

        if first_person_launch and (urls or launch_cta):
            score += 3
            reasons.append(
                "first-person product launch CTA"
            )

        # ---------------------------------------------------------
        # Commercial/product pitch structure
        #
        # Catch posts such as:
        # "Streamline deal-sourcing ... with our AI agents"
        # even when they contain no explicit "sign up" CTA.
        # ---------------------------------------------------------

        first_person = re.search(
            r"\b(?:we|our|my|i)\b",
            lowered,
        )

        product_noun = re.search(
            r"\b(?:platform|product|tool|app|service|solution|software|"
            r"saas|agents?|api|sdk)\b",
            lowered,
        )

        benefit_language = re.search(
            r"\b(?:streamline|automate|boost|supercharge|unlock|"
            r"accelerate|simplify|save|reduce|cut|improve|"
            r"scale|transform)\b",
            lowered,
        )

        if first_person and product_noun and benefit_language:
            score += 3
            reasons.append(
                "first-person product benefit pitch"
            )

        # ---------------------------------------------------------
        # Clamp score
        # ---------------------------------------------------------

        score = min(score, 10)

        # Remove duplicate reason strings while preserving order.
        reasons = list(dict.fromkeys(reasons))

        return PromotionResult(
            score=score,
            is_promotional=score >= self.threshold,
            reasons=reasons,
        )

