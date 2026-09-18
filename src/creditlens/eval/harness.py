"""Small eval harness: run each hand-labeled Q&A pair through the agent,
then use an LLM judge to score correctness and grounding.

Produces two numbers worth quoting:
  - accuracy: fraction of answers that match the gold answer in substance
  - hallucination_rate: fraction of answers with no tool-backed evidence at
    all (neither a quoted filing excerpt nor an as-of-dated ratio)
"""
from __future__ import annotations

import json
import os
import re
import sys
from statistics import mean

from dotenv import load_dotenv

from ..agent.factory import build_agent, has_key
from ..agent.tools import ToolRuntime
from ..edgar.client import EdgarClient
from ..rag.embeddings import embed_texts
from ..rag.vectorstore import FilingVectorStore

JUDGE_PROMPT = """You are grading an AI credit-research assistant's answer. Respond with ONLY a JSON object, no other text.

Question: {question}
Gold answer (ground truth): {gold_answer}
Assistant's answer: {answer}
Assistant's tool-backed evidence: {evidence}

Score:
- "correct": true if the assistant's answer matches the gold answer in substance \
(numbers within ~5%, same qualitative conclusion), false otherwise.
- "grounded": true if the assistant's claim is backed by at least one piece of the \
tool-backed evidence above (a quoted filing excerpt, or ratios computed as-of a \
specific filing date both count), false if it asserts facts with no tool backing at all.

Return exactly: {{"correct": true/false, "grounded": true/false, "reason": "<one sentence>"}}
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0) if match else text)


def _summarize_for_judge(evidence: list[dict]) -> list[str]:
    summaries = []
    for e in evidence:
        if "accessionNumber" in e:
            summaries.append(f"{e['form']} filed {e['filingDate']}: \"{e.get('excerpt', '')[:150]}\"")
        elif "as_of" in e:
            summaries.append(f"financial ratios computed as of {e['as_of']} (XBRL)")
    return summaries or ["none"]


def _build_judge():
    """Prefer Claude as judge for consistency; fall back to Gemini if that's
    the only key available."""
    if has_key("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic

        client = Anthropic()

        def judge(prompt: str) -> str:
            resp = client.messages.create(
                model="claude-sonnet-5", max_tokens=300, messages=[{"role": "user", "content": prompt}]
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return judge

    if has_key("GEMINI_API_KEY"):
        from google import genai

        from ..agent.gemini_backend import DEFAULT_MODEL, call_with_retry
        from ..agent.rate_limiter import RateLimiter

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        limiter = RateLimiter(max_calls=int(os.environ.get("GEMINI_RATE_LIMIT_PER_MIN", "60")))

        def judge(prompt: str) -> str:
            resp = call_with_retry(
                limiter, lambda: client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
            )
            return resp.text

        return judge

    raise RuntimeError("No usable API key found for the judge model.")


def run_eval(dataset_path: str) -> dict:
    load_dotenv()

    edgar = EdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT"))
    store = FilingVectorStore(persist_dir="data/chroma", collection_name="filings")
    runtime = ToolRuntime(edgar, store, embed_texts)
    agent = build_agent(runtime)
    judge = _build_judge()

    with open(dataset_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = []
    for case in cases:
        answer, evidence = agent.answer_question(case["ticker"], case["question"])
        prompt = JUDGE_PROMPT.format(
            question=case["question"],
            gold_answer=case["gold_answer"],
            answer=answer,
            evidence=_summarize_for_judge(evidence),
        )
        verdict = _extract_json(judge(prompt))
        results.append({**case, "answer": answer, "evidence": evidence, **verdict})
        print(f"[{'OK' if verdict['correct'] else 'MISS'}] {case['question']}")

    accuracy = mean(1.0 if r["correct"] else 0.0 for r in results)
    hallucination_rate = mean(0.0 if r["grounded"] else 1.0 for r in results)
    return {
        "n": len(results),
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "results": results,
    }


def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "src/creditlens/eval/dataset.jsonl"
    summary = run_eval(dataset_path)
    print()
    print(f"n={summary['n']}  accuracy={summary['accuracy']:.2f}  hallucination_rate={summary['hallucination_rate']:.2f}")


if __name__ == "__main__":
    main()
