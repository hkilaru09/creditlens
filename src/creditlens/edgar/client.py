"""Thin client for SEC EDGAR's public JSON APIs.

No API key required, but SEC requires a real contact identity in the
User-Agent header on every request, and asks callers to stay under ~10
requests/second. See https://www.sec.gov/os/webmaster-faq#developers.
"""
from __future__ import annotations

import time

import requests

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_MIN_REQUEST_INTERVAL = 0.15  # seconds, keeps us well under SEC's rate limit


class EdgarClient:
    def __init__(self, user_agent: str | None):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC EDGAR requires a User-Agent identifying a real contact, e.g. "
                "'CreditLens yourname@example.com'. Set SEC_EDGAR_USER_AGENT in .env."
            )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_at = 0.0
        self._ticker_map: dict[str, str] | None = None

    def _get(self, url: str) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        resp = self._session.get(url, timeout=15)
        self._last_request_at = time.monotonic()
        resp.raise_for_status()
        return resp

    def cik_for_ticker(self, ticker: str) -> str:
        if self._ticker_map is None:
            data = self._get(TICKER_MAP_URL).json()
            self._ticker_map = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
                for entry in data.values()
            }
        ticker = ticker.upper()
        if ticker not in self._ticker_map:
            raise KeyError(f"Ticker {ticker!r} not found in SEC's ticker list")
        return self._ticker_map[ticker]

    def submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def company_facts(self, cik: str) -> dict:
        return self._get(COMPANY_FACTS_URL.format(cik=cik)).json()

    def latest_filings(
        self, cik: str, form_types: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 4
    ) -> list[dict]:
        recent = self.submissions(cik)["filings"]["recent"]
        results = []
        for i, form in enumerate(recent["form"]):
            if form in form_types:
                results.append(
                    {
                        "form": form,
                        "accessionNumber": recent["accessionNumber"][i],
                        "filingDate": recent["filingDate"][i],
                        "reportDate": recent["reportDate"][i],
                        "primaryDocument": recent["primaryDocument"][i],
                    }
                )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
        accession_nodash = accession_number.replace("-", "")
        cik_nozero = str(int(cik))
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/"
            f"{accession_nodash}/{primary_document}"
        )

    def fetch_filing_html(self, cik: str, accession_number: str, primary_document: str) -> str:
        url = self.filing_document_url(cik, accession_number, primary_document)
        return self._get(url).text
