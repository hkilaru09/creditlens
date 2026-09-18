"""Small eval harness: run each hand-labeled Q&A pair through the agent,
then use Claude as a judge to score correctness and grounding.

Produces two numbers worth quoting:
  - accuracy: fraction of answers that match the gold answer in substance
  - hallucination_rate: fraction of answers NOT backed by a specific filing citation
"""
from __future__ import annotations

import json
import os
import sys
from statistics import mean

from anthropic import Anthropic
from dotenv import load_dotenv

from ..agent.orchestrator import Agent
from ..agent.tools import ToolRuntime
from ..edgar.client import EdgarClient
from ..rag.embeddings import embed_texts
from ..rag.vectorstore import FilingVectorStore

JUDGE_PROMPT = """You are grading an AI credit-research assistant's answer. Respond with ONLY a JSON object, no other text.

Question: {question}
Gold answer (ground truth): {gold_answer}
Assistant's answer: {answer}
Assistant's cited sources: {citations}

Score:
- "correct": true if the assistant's answer matches the gold answer in substance \
(numbers within ~5%, same qualitative conclusion), false otherwise.
- "grounded": true if the assistant cites a specific filing (form + filing date or \
accession number) to support its claim, false if it asserts facts with no citation.

Return exactly: {{"correct": true/false, "grounded": true/false, "reason": "<one sentence>"}}
"""


def _judge(client: Anthropic, question: str, gold_answer: str, answer: str, citations: list[dict]) -> dict:
    citation_summary = [f"{c['form']} filed {c['filingDate']}" for c in citations] or ["none"]
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    answer=answer,
                    citations=citation_summary,
                ),
            }
        ],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def run_eval(dataset_path: str) -> dict:
    load_dotenv()
    client = Anthropic()

    edgar = EdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT"))
    store = FilingVectorStore(persist_dir="data/chroma", collection_name="filings")
    runtime = ToolRuntime(edgar, store, embed_texts)
    agent = Agent(runtime, client=client)

    with open(dataset_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = []
    for case in cases:
        answer, citations = agent.answer_question(case["ticker"], case["question"])
        verdict = _judge(client, case["question"], case["gold_answer"], answer, citations)
        results.append({**case, "answer": answer, "citations": citations, **verdict})
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
