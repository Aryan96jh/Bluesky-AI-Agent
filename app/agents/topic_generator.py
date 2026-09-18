class TopicGenerator:

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate_topics(self) -> list[str]:

        return [
            "AI agents",
            "AI agent benchmark",
            "AI agent evaluation",
            "AI agent framework",
            "AI agent memory",
            "AI agent tool use",
            "AI agent planning",
            "AI agent reliability",
            "AI agent failure",
            "AI agent architecture",

            "LLM benchmark",
            "LLM evaluation",
            "LLM inference",
            "LLM reasoning",
            "LLM tool calling",
            "LLM context window",

            "AI coding agent",
            "coding agent benchmark",
            "coding agent evaluation",
            "AI developer tools",

            "MCP AI agents",
            "MCP tools",

            "agentic workflow",
            "AI automation workflow",

            "AI safety",
            "AI alignment",
            "AI governance",
            "AI infrastructure",
            "open source AI",
            "AI security",

            "AI agents experiment",
            "AI agents production",
        ]
