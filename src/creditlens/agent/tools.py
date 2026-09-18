"""Tool schemas exposed to Claude, and the runtime that executes them.

This is the "MCP-shaped" part of the system: a fixed set of callable tools
(fetch filings, compute ratios, search indexed text) that the model chains
on its own. It's implemented as native Claude tool-use rather than a
separate MCP server process, since the whole app runs as one script.
"""
from __future__ import annotations

from ..edgar.client import EdgarClient
from ..edgar.xbrl import compute_ratios
from ..rag.vectorstore import FilingVectorStore

TOOLS = [
    {
        "name": "list_recent_filings",
        "description": (
            "List a company's most recent 10-K/10-Q filings with filing dates "
            "and SEC accession numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financial_ratios",
        "description": (
            "Compute headline credit ratios (current ratio, debt/equity, "
            "debt/EBITDA approx, interest coverage) from the company's most "
            "recent 10-K XBRL data. Use this for any quantitative leverage or "
            "liquidity claim instead of estimating from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "search_filings",
        "description": (
            "Semantic search over the company's indexed 10-K/10-Q text. Returns "
            "the most relevant excerpts, each tagged with its form, filing date, "
            "and accession number so it can be cited. Use this for qualitative "
            "claims (risk factors, covenant language, MD&A commentary)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "query": {"type": "string", "description": "What to search for"},
                "n_results": {"type": "integer", "default": 5},
            },
            "required": ["ticker", "query"],
        },
    },
]


class ToolRuntime:
    def __init__(self, edgar: EdgarClient, store: FilingVectorStore, embed_fn):
        self.edgar = edgar
        self.store = store
        self.embed_fn = embed_fn
        self._cik_cache: dict[str, str] = {}

    def _cik(self, ticker: str) -> str:
        ticker = ticker.upper()
        if ticker not in self._cik_cache:
            self._cik_cache[ticker] = self.edgar.cik_for_ticker(ticker)
        return self._cik_cache[ticker]

    def run(self, name: str, tool_input: dict) -> dict:
        if name == "list_recent_filings":
            cik = self._cik(tool_input["ticker"])
            return {"filings": self.edgar.latest_filings(cik)}

        if name == "get_financial_ratios":
            cik = self._cik(tool_input["ticker"])
            facts = self.edgar.company_facts(cik)
            return compute_ratios(facts)

        if name == "search_filings":
            [embedding] = self.embed_fn([tool_input["query"]])
            result = self.store.query(
                embedding,
                n_results=tool_input.get("n_results", 5),
                where={"ticker": tool_input["ticker"].upper()},
            )
            documents = result.get("documents") or [[]]
            metadatas = result.get("metadatas") or [[]]
            citations = [
                {"excerpt": doc, **meta}
                for doc, meta in zip(documents[0], metadatas[0])
            ]
            return {"citations": citations}

        raise ValueError(f"Unknown tool: {name}")
