"""Financial situation memory — dual backend.

Phase 1 (v0.2.x): BM25 vector memory for lexical matching.
Phase 2 (v0.3.x): Append-only markdown decision log (TradingMemoryLog) from upstream.

This module provides:
- FinancialSituationMemory: BM25 backend (kept for bull/bear/trader agents that use get_memories)
- TradingMemoryLog: Append-only markdown log (used by portfolio manager for past_context)
"""

import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections.

    Ported from upstream v0.3.1.
    """
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = cfg.get("memory_log_max_entries")

    def store_decision(self, ticker: str, trade_date: str, final_trade_decision: str) -> None:
        if not self._log_path:
            return
        from tradingagents.agents.utils.rating import parse_rating
        if self._log_path.exists():
            raw = self._log_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("| pending]"):
                    return
        rating = parse_rating(final_trade_decision)
        tag = f"[{trade_date} | {ticker} | {rating} | pending]"
        entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def load_entries(self) -> list:
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            first_line = raw.splitlines()[0] if raw.splitlines() else ""
            m = re.match(
                r"\[(\d{4}-\d{2}-\d{2})\s*\|\s*(\S+)\s*\|\s*(\w+)\s*\|\s*(\w+)\]",
                first_line,
            )
            if not m:
                continue
            date_str, ticker, rating, status = m.groups()
            dec_m = self._DECISION_RE.search(raw)
            ref_m = self._REFLECTION_RE.search(raw)
            entries.append({
                "date": date_str,
                "ticker": ticker,
                "rating": rating,
                "status": status,
                "decision": dec_m.group(1).strip() if dec_m else "",
                "reflection": ref_m.group(1).strip() if ref_m else "",
            })
        return entries

    def get_pending_entries(self) -> list:
        return [e for e in self.load_entries() if e["status"] == "pending"]

    def get_past_context(self, company_name: str, limit: int = 5) -> str:
        entries = self.load_entries()
        relevant = [e for e in entries if e["status"] == "resolved"]
        if not relevant:
            return ""
        recent = relevant[-limit:]
        parts = []
        for e in recent:
            part = f"[{e['date']} | {e['ticker']}] Rating: {e['rating']}"
            if e.get("reflection"):
                part += f" — Reflection: {e['reflection'][:200]}"
            parts.append(part)
        return "\n".join(parts)

    def batch_update_with_outcomes(self, updates: list) -> None:
        if not self._log_path or not self._log_path.exists():
            return
        text = self._log_path.read_text(encoding="utf-8")
        for ticker, trade_date, outcome in updates:
            tag = f"[{trade_date} | {ticker} |"
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith(tag) and "| pending]" in line:
                    lines[i] = line.replace("| pending]", f"| resolved]")
                    break
            text = "\n".join(lines)
        self._log_path.write_text(text, encoding="utf-8")


class FinancialSituationMemory:
    """BM25-based memory for bull/bear/trader agents.

    Kept for backward compatibility — agents still call get_memories().
    """
    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _rebuild_index(self):
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)

    def add_situation(self, situation: str, recommendation: str):
        self.documents.append(situation)
        self.recommendations.append(recommendation)
        self._rebuild_index()

    def add_situations(self, situations: list):
        for s in situations:
            if isinstance(s, dict):
                self.add_situation(s.get("situation", ""), s.get("recommendation", ""))
            else:
                self.add_situation(str(s), "")

    def get_memories(self, situation: str, n_matches: int = 3) -> list:
        if not self.bm25 or not self.documents:
            return []
        query_tokens = self._tokenize(situation)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]
        return [
            {"recommendation": self.recommendations[i], "score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]
