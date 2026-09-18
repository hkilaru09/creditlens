# CreditLens

An automated credit-research agent. Give it a ticker; it pulls the company's
SEC filings, computes headline credit ratios from XBRL data, retrieves
relevant filing text, and drafts a one-page credit memo — citing every claim
back to a specific filing.

Built to automate the "analyst grunt work" of pulling filings and drafting a
first-pass risk memo before real analysis starts.

## Architecture

- **`edgar/`** — SEC EDGAR client (no API key; SEC just requires a contact
  email in the User-Agent header) + XBRL parser that computes current ratio,
  debt/equity, debt/EBITDA (approx), and interest coverage directly from a
  company's tagged financial facts.
- **`rag/`** — chunks filing HTML into plain-text windows, embeds them
  locally with `sentence-transformers` (no API key), and stores them in a
  local Chroma DB. Retrieval returns excerpts tagged with form/date/accession
  number for citation.
- **`agent/`** — a tool-use loop with three tools (`list_recent_filings`,
  `get_financial_ratios`, `search_filings`) that the model chains on its own
  to draft the memo or answer a specific question. This is the same shape of
  tool-calling MCP formalizes, implemented natively since the whole thing
  runs as one script rather than a standalone server. Two interchangeable
  backends implement the same `draft_memo`/`answer_question` interface:
  `orchestrator.Agent` (Claude, `anthropic` SDK) and
  `gemini_backend.GeminiAgent` (Gemini, `google-genai` SDK, with a
  self-imposed rate limiter and retry/backoff on 429/503s since free-tier
  Gemini quotas can be very low per model). `agent/factory.py` picks one
  based on which API key is set, or `LLM_PROVIDER=anthropic|gemini`.
- **`eval/`** — a small hand-labeled Q&A benchmark (`dataset.jsonl`) scored
  by an LLM-judge pass for **accuracy** (does the answer match the gold
  answer) and **hallucination rate** (is every claim backed by a citation).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in ANTHROPIC_API_KEY and SEC_EDGAR_USER_AGENT
```

`SEC_EDGAR_USER_AGENT` just needs to look like `"YourApp yourname@email.com"`
— SEC requires a real contact identity on every request, no registration.

## Usage

```bash
# Draft a full credit memo (indexes filings on first run)
creditlens AAPL

# Ask a specific question instead
creditlens AAPL --question "What's driving the leverage increase?"

# Force re-fetch and re-index filings
creditlens AAPL --reindex --filings 4
```

## Eval

`src/creditlens/eval/dataset.jsonl` ships with 3 seed examples (AAPL,
verified against live EDGAR data). For a real benchmark, hand-label more
pairs by reading a few filings yourself and writing the gold answers —
that's the part that makes the eval numbers meaningful rather than
circular.

```bash
python -m creditlens.eval.harness src/creditlens/eval/dataset.jsonl
```

Reports `accuracy` (gold-answer match) and `hallucination_rate` (fraction of
answers backed by no tool evidence at all — a quoted filing excerpt from
`search_filings` and an as-of-dated ratio from `get_financial_ratios` both
count as grounded; only an unsupported assertion counts as a hallucination).

## Notes / next steps

- Ratios are computed only from near-universal `us-gaap` tags (`Assets`,
  `LiabilitiesCurrent`, `StockholdersEquity`, ...). Some companies tag things
  slightly differently; a missing ratio means the tag wasn't found, not that
  it's zero.
- `data/chroma/` and `data/filings/` are gitignored — they're a local cache,
  regenerate them anytime with `--reindex`.
- If you want a literal MCP server (for use from Claude Desktop rather than
  this CLI), `agent/tools.py`'s `ToolRuntime` is already the right shape —
  it just needs an MCP stdio/SSE transport wrapped around it.
