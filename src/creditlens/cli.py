import argparse
import os

from dotenv import load_dotenv

from .agent.factory import build_agent
from .agent.tools import ToolRuntime
from .edgar.client import EdgarClient
from .rag.embeddings import embed_texts
from .rag.index import index_filing
from .rag.vectorstore import FilingVectorStore


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="CreditLens: automated credit-research agent")
    parser.add_argument("ticker")
    parser.add_argument("--question", help="Ask one specific question instead of drafting a memo")
    parser.add_argument(
        "--reindex", action="store_true", help="Re-fetch and re-index filings before running"
    )
    parser.add_argument(
        "--filings", type=int, default=2, help="Number of recent 10-K/10-Q filings to index"
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    edgar = EdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT"))
    store = FilingVectorStore(persist_dir="data/chroma", collection_name="filings")

    cik = edgar.cik_for_ticker(ticker)

    existing = store.query(embed_texts([ticker])[0], n_results=1, where={"ticker": ticker})
    already_indexed = bool(existing.get("ids") and existing["ids"][0])

    if args.reindex or not already_indexed:
        for filing in edgar.latest_filings(cik, limit=args.filings):
            n = index_filing(store, edgar, ticker, cik, filing)
            print(f"Indexed {n} chunks from {filing['form']} filed {filing['filingDate']}")

    runtime = ToolRuntime(edgar, store, embed_texts)
    agent = build_agent(runtime)

    if args.question:
        answer, citations = agent.answer_question(ticker, args.question)
    else:
        answer, citations = agent.draft_memo(ticker)

    print()
    print(answer)
    if citations:
        print("\n--- sources cited ---")
        seen = set()
        for c in citations:
            key = (c["form"], c["filingDate"], c["accessionNumber"])
            if key not in seen:
                seen.add(key)
                print(f"- {c['form']} filed {c['filingDate']} (accession {c['accessionNumber']})")


if __name__ == "__main__":
    main()
