from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _extract_cost(resp: Any) -> dict[str, Any]:
    """Pull cost and token usage out of a litellm response object."""
    cost = 0.0
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost", 0) or 0.0

    usage: dict[str, int] = {}
    resp_usage = getattr(resp, "usage", None)
    if resp_usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(resp_usage, key, None)
            if val is None and isinstance(resp_usage, dict):
                val = resp_usage.get(key)
            if val is not None:
                usage[key] = int(val)

    return {"cost_usd": float(cost), "usage": usage}


@dataclass
class CostTracker:
    """Accumulates cost/token data from litellm calls."""

    _calls: list[dict[str, Any]] = field(default_factory=list)

    def track(self, category: str, resp: Any) -> float:
        """Record one LLM/embedding call. Returns the cost in USD."""
        info = _extract_cost(resp)
        info["category"] = category
        self._calls.append(info)
        return info["cost_usd"]

    @property
    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self._calls)

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, Any]] = {}
        for call in self._calls:
            cat = call["category"]
            if cat not in by_category:
                by_category[cat] = {
                    "calls": 0,
                    "cost_usd": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            by_category[cat]["calls"] += 1
            by_category[cat]["cost_usd"] += call["cost_usd"]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                by_category[cat][k] += call["usage"].get(k, 0)

        return {
            "total_cost_usd": self.total_cost,
            "total_calls": self.total_calls,
            "by_category": by_category,
        }


class RunLogger:
    """Logs benchmark run events to a structured JSON file.

    Each run produces one JSON file containing run parameters, loaded samples,
    Q&A interactions, and scoring results (optionally including raw judge calls).
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        log_judge_calls: bool = False,
        run_params: dict[str, Any] | None = None,
        run_name: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_judge_calls = log_judge_calls

        self._run_id = uuid.uuid4().hex[:12]
        self._run_name = run_name
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._started_at_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._run_params = run_params or {}
        self._samples: list[dict[str, Any]] = []
        self._current_sample: dict[str, Any] | None = None
        self._question_counter = 0
        self.cost_tracker = cost_tracker

    @property
    def run_id(self) -> str:
        return self._run_id

    def update_params(self, extra: dict[str, Any]) -> None:
        """Merge additional key-value pairs into the run parameters."""
        self._run_params.update(extra)

    def log_sample_loaded(
        self,
        sample_idx: int,
        original_sample_idx: int | None,
        factual_state: str,
        atomic_facts: list[str],
    ) -> None:
        self._finalize_current_sample()
        self._question_counter = 0
        self._current_sample = {
            "sample_idx": sample_idx,
            "original_sample_idx": original_sample_idx,
            "factual_state": factual_state,
            "atomic_facts": atomic_facts,
            "events": [],
        }

    def log_question(self, question: str, answer: str) -> None:
        if self._current_sample is None:
            return
        self._question_counter += 1
        self._current_sample["events"].append({
            "type": "question",
            "round": self._question_counter,
            "question": question,
            "answer": answer,
        })

    def log_score(
        self,
        score_result: dict[str, Any],
        judge_details: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._current_sample is None:
            return
        event: dict[str, Any] = {
            "type": "score",
            "round": self._question_counter,
            "result": {
                "coverage_ratio": score_result.get("coverage_ratio"),
                "original_facts": score_result.get("original_facts"),
                "covered_facts": score_result.get("covered_facts"),
                "questions_asked": score_result.get("questions_asked"),
                "answers_collected": score_result.get("answers_collected"),
                "question_quality": score_result.get("question_quality"),
                "no_document_answers": score_result.get("no_document_answers"),
                "total_questions": score_result.get("total_questions"),
            },
        }
        if self._log_judge_calls and judge_details:
            event["judge_calls"] = judge_details
        self._current_sample["events"].append(event)

    def _get_log_path(self) -> Path:
        """Return the log file path for this run (fixed at start)."""
        slug = f"_{self._run_name}" if self._run_name else ""
        slug = slug.replace("/", "-").replace("\\", "-")
        return self._log_dir / f"run_{self._started_at_ts}{slug}_{self._run_id}.json"

    def _persist(self, *, finished: bool = False) -> None:
        """Write current run state to disk. Called after each sample and on flush."""
        log_data: dict[str, Any] = {
            "run_id": self._run_id,
        }
        if self._run_name:
            log_data["run_name"] = self._run_name
        log_data["started_at"] = self._started_at
        if finished:
            log_data["finished_at"] = datetime.now(timezone.utc).isoformat()
        log_data["parameters"] = self._run_params
        log_data["samples"] = list(self._samples)
        if self.cost_tracker is not None:
            log_data["cost_summary"] = self.cost_tracker.summary()
        path = self._get_log_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def flush(self) -> Path:
        """Finalize the run, write the log to disk, and return the file path."""
        self._finalize_current_sample()
        self._persist(finished=True)
        return self._get_log_path()

    def restore_samples(self, samples: list[dict[str, Any]]) -> None:
        """Import already-computed samples from a previous run.

        These are inserted at the beginning of the samples list so that later
        ``flush()`` writes them alongside newly computed ones.
        """
        self._samples = list(samples) + self._samples

    @property
    def restored_sample_indices(self) -> set[int]:
        """Return the set of ``sample_idx`` values already present."""
        return {s["sample_idx"] for s in self._samples}

    def _finalize_current_sample(self) -> None:
        if self._current_sample is not None:
            self._samples.append(self._current_sample)
            self._current_sample = None
            self._persist(finished=False)
