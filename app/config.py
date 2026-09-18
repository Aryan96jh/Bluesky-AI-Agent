import os

from dotenv import load_dotenv

load_dotenv()


BOT_NAME = os.getenv(
    "BOT_NAME",
    "AI Observer",
)


BOT_NICHE = os.getenv(
    "BOT_NICHE",
    "AI agents, autonomous AI systems, LLMs and AI automation",
)


NICHE = BOT_NICHE


BOT_TONE = os.getenv(
    "BOT_TONE",
    "casual, knowledgeable, slightly witty",
)
