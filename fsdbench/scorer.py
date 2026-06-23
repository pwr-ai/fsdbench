from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import litellm

if TYPE_CHECKING:
    from .event_logger import CostTracker

from .agent import NO_DOCUMENT_ANSWER

JUDGE_SYSTEM_PROMPT = """
Jesteś audytorem pokrycia informacji.

Masz:
- ZDANIE_ORYGINALNE: jedno zdanie (źródło prawdy).
- ZDANIA_PRZYPASOWANE: lista zdań, które mogą zawierać te same informacje, ale mogą \
też zawierać dodatkowe informacje o innych sprawach.

Zadanie:
Oceń, czy ZDANIA_PRZYPASOWANE (rozpatrywane łącznie) zawierają WSZYSTKIE informacje \
z ZDANIA_ORYGINALNEGO.

Definicja "zawierają wszystkie informacje":
- Każdy fakt z ZDANIA_ORYGINALNEGO (kto/co/komu/co się dzieje/warunki) musi wystąpić \
w ZDANIACH_PRZYPASOWANYCH wprost albo jako jednoznaczny ekwiwalent językowy.
- Dodatkowe fakty w ZDANIACH_PRZYPASOWANYCH są dozwolone i nie obniżają oceny \
(nie powodują decyzji "Nie").
- Jeśli JAKIEKOLWIEK pojedyncze zdanie w ZDANIACH_PRZYPASOWANYCH przekazuje dokładnie \
ten sam sens co ZDANIE_ORYGINALNE, to odpowiedź musi być "Tak" (nawet jeśli wśród \
pozostałych zdań są inne, niepowiązane informacje).
- Nie wolno dopowiadać ani domyślać: jeśli choć jeden fakt z oryginału nie jest \
wyrażony wprost, decyzja to "Nie".

Odpowiedź:
Zwróć WYŁĄCZNIE JSON dokładnie w formacie:
{"decision":"Tak"} albo {"decision":"Nie"}
Bez żadnego dodatkowego tekstu.
""".strip()


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(norms_a > 0, a / norms_a, 0.0)
    b_norm = np.where(norms_b > 0, b / norms_b, 0.0)
    return a_norm @ b_norm.T


def _parse_judge_decision(text: str) -> bool:
    t = (text or "").strip().lower()
    if "decision" in t:
        if '"tak"' in t or "'tak'" in t:
            return True
        if '"nie"' in t or "'nie'" in t:
            return False
    if t.startswith("tak"):
        return True
    return False


@dataclass
class SemanticScorer:
    """Scores how many atomic facts were discovered using embedding similarity + litellm judge."""

    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 512
    top_k: int = 20
    verbose: bool = False
    embedding_cache: dict[str, np.ndarray] | None = None
    cost_tracker: CostTracker | None = None

    def __post_init__(self) -> None:
        if self.embedding_cache is None:
            self.embedding_cache = {}

    def score(
        self,
        atomic_facts: list[str],
        discovered_texts: list[str],
    ) -> dict[str, Any]:
        """Evaluate what fraction of atomic facts is covered by discovered texts."""
        if not atomic_facts:
            return self._empty_result()

        mapping = self._match_facts(atomic_facts, discovered_texts)
        return self._evaluate(atomic_facts, mapping)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            cached = self.embedding_cache.get(key)
            if cached is not None:
                vectors[i] = cached
            else:
                to_fetch.append((i, text))

        for start in range(0, len(to_fetch), self.embedding_batch_size):
            batch = to_fetch[start : start + self.embedding_batch_size]
            resp = litellm.embedding(
                model=self.embedding_model, input=[t for _, t in batch]
            )
            if self.cost_tracker is not None:
                self.cost_tracker.track("embedding", resp)
            for j, item in enumerate(resp.data):
                vec = np.array(item["embedding"], dtype=np.float32)
                idx, text = batch[j]
                vectors[idx] = vec
                self.embedding_cache[hashlib.sha256(text.encode()).hexdigest()] = vec

        return np.stack(vectors)

    def _match_facts(
        self,
        facts: list[str],
        discovered: list[str],
    ) -> dict[int, list[tuple[str, float]]]:
        """For each fact, find the top-k most similar discovered texts."""
        mapping: dict[int, list[tuple[str, float]]] = {
            i: [] for i in range(len(facts))
        }

        valid = [
            d.strip()
            for d in discovered
            if d.strip() and d.strip().lower() != NO_DOCUMENT_ANSWER.lower()
        ]
        if not valid:
            return mapping

        all_texts = list(facts) + valid
        all_vecs = self._embed(all_texts)
        fact_vecs = all_vecs[: len(facts)]
        disc_vecs = all_vecs[len(facts) :]

        sim = _cosine_similarity_matrix(fact_vecs, disc_vecs)

        for d_idx, d_text in enumerate(valid):
            ranked = sorted(
                ((i, float(sim[i, d_idx])) for i in range(len(facts))),
                key=lambda x: x[1],
                reverse=True,
            )
            for fact_idx, similarity in ranked[: self.top_k]:
                mapping[fact_idx].append((d_text, similarity))

        for i in mapping:
            mapping[i] = sorted(
                mapping[i], key=lambda x: x[1], reverse=True
            )[: self.top_k]

        return mapping

    def _judge_coverage(
        self, original: str, candidates: list[str]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return (covered, judge_detail_or_None)."""
        if not candidates:
            return False, None

        user_prompt = (
            "ZDANIE_ORYGINALNE:\n"
            f"{original}\n\n"
            "ZDANIA_PRZYPASOWANE:\n"
            + "\n".join(f"- {s}" for s in candidates)
        )

        if self.verbose:
            print(f"[JUDGE] {original[:80]}...")
            print(
                f"  Matched ({len(candidates)}): "
                f"{[s[:50] + '...' for s in candidates]}"
            )

        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.cost_tracker is not None:
            self.cost_tracker.track("judge", resp)

        raw_response = resp.choices[0].message.content or ""
        decision = _parse_judge_decision(raw_response)

        if self.verbose:
            print(f"  Decision: {raw_response}")

        detail: dict[str, Any] = {
            "fact": original,
            "candidates": candidates,
            "decision": decision,
            "raw_response": raw_response,
        }
        return decision, detail

    def _evaluate(
        self,
        facts: list[str],
        mapping: dict[int, list[tuple[str, float]]],
    ) -> dict[str, Any]:
        details = []
        undiscovered = []
        judge_details: list[dict[str, Any]] = []

        for i, fact in enumerate(facts):
            candidates = [text for text, _ in mapping.get(i, [])]
            covered, judge_detail = self._judge_coverage(fact, candidates)
            details.append({
                "idx": i,
                "text": fact,
                "covered": covered,
                "matched_to": candidates,
            })
            if judge_detail is not None:
                judge_details.append(judge_detail)
            if not covered:
                undiscovered.append(fact)

        return {
            "coverage_ratio": 1.0 - len(undiscovered) / len(facts),
            "original_facts": len(facts),
            "covered_facts": len(facts) - len(undiscovered),
            "undiscovered_facts": undiscovered,
            "facts": details,
            "judge_details": judge_details,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "coverage_ratio": 1.0,
            "original_facts": 0,
            "covered_facts": 0,
            "undiscovered_facts": [],
            "facts": [],
        }
