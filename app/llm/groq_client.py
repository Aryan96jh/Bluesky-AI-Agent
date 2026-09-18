import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq


load_dotenv(dotenv_path=".env")


class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")

        self.client = Groq(api_key=api_key)

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        print(f"[GROQ] Model: {self.model}")
        print("[GROQ] Reasoning effort: low")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.2,
    ) -> str:

        max_tokens = max(
            150,
            min(int(max_tokens), 600),
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort="low",
                )

                choice = response.choices[0]
                message = choice.message

                content = message.content or ""

                finish_reason = getattr(
                    choice,
                    "finish_reason",
                    None,
                )

                reasoning = getattr(
                    message,
                    "reasoning",
                    None,
                ) or ""

                print(
                    f"[GROQ] finish_reason={finish_reason} "
                    f"content={len(content)} "
                    f"reasoning={len(reasoning)}"
                )

                if content.strip():
                    return content.strip()

                print(
                    "[GROQ] Empty content returned. "
                    f"Attempt {attempt + 1}/3"
                )

                if attempt < 2:
                    time.sleep(1)
                    continue

                raise RuntimeError(
                    "Groq returned empty content."
                )

            except Exception as e:
                error_text = str(e)

                print(
                    f"[GROQ] Error on attempt "
                    f"{attempt + 1}/3: {error_text}"
                )

                if (
                    "429" in error_text
                    or "rate limit" in error_text.lower()
                    or "too many requests" in error_text.lower()
                ):
                    if attempt < 2:
                        wait_time = 2 * (attempt + 1)

                        print(
                            f"[GROQ] Rate limited. "
                            f"Waiting {wait_time}s..."
                        )

                        time.sleep(wait_time)
                        continue

                if attempt < 2:
                    time.sleep(1)
                    continue

                raise

        raise RuntimeError(
            "Groq generation failed."
        )

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.1,
    ) -> dict:

        max_tokens = max(
            250,
            min(int(max_tokens), 700),
        )

        system_prompt = (
            "You are a JSON classifier. "
            "Return ONLY one valid JSON object. "
            "No markdown. "
            "No explanation. "
            "No comments. "
            "Keep the JSON extremely short."
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort="low",
                )

                choice = response.choices[0]
                message = choice.message

                content = message.content or ""

                finish_reason = getattr(
                    choice,
                    "finish_reason",
                    None,
                )

                reasoning = getattr(
                    message,
                    "reasoning",
                    None,
                ) or ""

                print(
                    f"[GROQ JSON] "
                    f"finish_reason={finish_reason} "
                    f"content={len(content)} "
                    f"reasoning={len(reasoning)}"
                )

                if not content.strip():
                    print(
                        "[GROQ JSON] Empty content. "
                        f"Attempt {attempt + 1}/3"
                    )

                    if attempt < 2:
                        time.sleep(1)
                        continue

                    raise RuntimeError(
                        "Groq returned empty JSON content."
                    )

                parsed = self._parse_json(content)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "Groq returned JSON but it was "
                        "not an object."
                    )

                return parsed

            except Exception as e:
                error_text = str(e)

                print(
                    f"[GROQ JSON] Error on attempt "
                    f"{attempt + 1}/3: {error_text}"
                )

                if (
                    "429" in error_text
                    or "rate limit" in error_text.lower()
                    or "too many requests" in error_text.lower()
                ):
                    if attempt < 2:
                        wait_time = 2 * (attempt + 1)

                        print(
                            f"[GROQ JSON] Rate limited. "
                            f"Waiting {wait_time}s..."
                        )

                        time.sleep(wait_time)
                        continue

                if attempt < 2:
                    time.sleep(1)
                    continue

                raise

        raise RuntimeError(
            "Groq JSON generation failed."
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()

        # Direct JSON.
        try:
            result = json.loads(text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Remove markdown code fences.
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

        try:
            result = json.loads(cleaned)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Extract JSON object from surrounding text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end > start:
            candidate = cleaned[
                start:end + 1
            ]

            try:
                result = json.loads(candidate)

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Could not parse Groq response as JSON: "
            f"{text[:500]}"
        )
