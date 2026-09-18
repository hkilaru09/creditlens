"""The agent loop: Claude + tools, chained until it produces a final answer."""
from __future__ import annotations

import json

from anthropic import Anthropic

from .memo import MEMO_SYSTEM_PROMPT, QA_SYSTEM_PROMPT
from .tools import TOOLS, ToolRuntime, summarize_evidence

MODEL = "claude-sonnet-5"
MAX_TOOL_TURNS = 6


class Agent:
    def __init__(self, runtime: ToolRuntime, client: Anthropic | None = None):
        self.runtime = runtime
        self.client = client or Anthropic()

    def _run_tool_loop(self, system_prompt: str, user_message: str) -> tuple[str, list[dict]]:
        messages = [{"role": "user", "content": user_message}]
        all_evidence: list[dict] = []

        for _ in range(MAX_TOOL_TURNS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return final_text, all_evidence

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self.runtime.run(block.name, block.input)
                all_evidence.extend(summarize_evidence(block.name, block.input, result))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return "Ran out of tool-call turns before reaching a final answer.", all_evidence

    def draft_memo(self, ticker: str) -> tuple[str, list[dict]]:
        return self._run_tool_loop(MEMO_SYSTEM_PROMPT, f"Draft a one-page credit memo for {ticker}.")

    def answer_question(self, ticker: str, question: str) -> tuple[str, list[dict]]:
        return self._run_tool_loop(QA_SYSTEM_PROMPT, f"Company: {ticker}\nQuestion: {question}")
