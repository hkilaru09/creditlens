"""Gemini backend for the same agent loop shape as orchestrator.Agent.

Kept as a separate class rather than folded into one abstraction: Claude and
Gemini structure tool-calling conversation history differently enough
(content-block roles, function_call vs tool_use, response wrapping) that a
shared interface would just be a thin wrapper hiding two incompatible
message formats. Both classes expose the same draft_memo/answer_question
methods so the CLI and eval harness can swap between them.
"""
from __future__ import annotations

import os
import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .memo import MEMO_SYSTEM_PROMPT, QA_SYSTEM_PROMPT
from .rate_limiter import RateLimiter
from .tools import TOOLS, ToolRuntime, summarize_evidence

DEFAULT_MODEL = "gemini-flash-lite-latest"
MAX_TOOL_TURNS = 6
DEFAULT_RATE_LIMIT_PER_MIN = 60
MAX_RETRIES = 5


def _retry_delay_seconds(error: Exception, fallback: float) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 0.5 if match else fallback


def call_with_retry(rate_limiter: RateLimiter, fn):
    """Runs fn() under the given rate limiter, retrying on transient 503s
    and quota 429s (backing off by the server's suggested retry delay)."""
    for attempt in range(MAX_RETRIES):
        rate_limiter.acquire()
        try:
            return fn()
        except genai_errors.ServerError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) != 429 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_retry_delay_seconds(e, fallback=2**attempt))


def _to_gemini_tool() -> types.Tool:
    return types.Tool(
        function_declarations=[
            {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
            for t in TOOLS
        ]
    )


class GeminiAgent:
    def __init__(
        self,
        runtime: ToolRuntime,
        client: genai.Client | None = None,
        model: str = DEFAULT_MODEL,
        rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
    ):
        self.runtime = runtime
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self._tool = _to_gemini_tool()
        self._limiter = RateLimiter(max_calls=rate_limit_per_min, period_seconds=60.0)

    def _generate(self, system_prompt: str, contents: list[types.Content]):
        return call_with_retry(
            self._limiter,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt, tools=[self._tool]
                ),
            ),
        )

    def _run_tool_loop(self, system_prompt: str, user_message: str) -> tuple[str, list[dict]]:
        contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
        all_evidence: list[dict] = []

        for _ in range(MAX_TOOL_TURNS):
            response = self._generate(system_prompt, contents)
            candidate = response.candidates[0]
            contents.append(candidate.content)

            function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
            if not function_calls:
                return response.text or "", all_evidence

            response_parts = []
            for fc in function_calls:
                fc_input = dict(fc.args)
                result = self.runtime.run(fc.name, fc_input)
                all_evidence.extend(summarize_evidence(fc.name, fc_input, result))
                response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
            contents.append(types.Content(role="user", parts=response_parts))

        return "Ran out of tool-call turns before reaching a final answer.", all_evidence

    def draft_memo(self, ticker: str) -> tuple[str, list[dict]]:
        return self._run_tool_loop(MEMO_SYSTEM_PROMPT, f"Draft a one-page credit memo for {ticker}.")

    def answer_question(self, ticker: str, question: str) -> tuple[str, list[dict]]:
        return self._run_tool_loop(QA_SYSTEM_PROMPT, f"Company: {ticker}\nQuestion: {question}")
