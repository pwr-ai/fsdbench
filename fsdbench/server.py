from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .agent import NO_DOCUMENT_ANSWER, FactChatAgent
from .event_logger import CostTracker, RunLogger
from .scorer import SemanticScorer

# Repo root is one level above the package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _REPO_ROOT / "data" / "raw-easy.json"


class BenchmarkServer:
    """
    Orchestrates factual-state discovery benchmarking.

    Loads a dataset of samples (factual_state + atomic_facts), exposes a QA
    agent for each sample, and scores discovery coverage.
    """

    def __init__(
        self,
        dataset_path: str | Path = _DEFAULT_DATASET,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        log_dir: str | Path | None = None,
        log_judge_calls: bool = False,
        run_name: str | None = None,
    ) -> None:
        self.model = model
        self.verbose = verbose
        self._samples = self._load_dataset(Path(dataset_path))
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.cost_tracker = CostTracker()

        self._sample_idx: int | None = None
        self.factual_state: str | None = None
        self._atomic_facts: list[str] | None = None
        self._agent: FactChatAgent | None = None
        self._answers: list[str] = []
        self._all_answers: list[str] = []
        self._covered_indices: set[int] = set()
        self._last_scored_answer_count: int = 0

        self._logger: RunLogger | None = None
        if log_dir is not None:
            self._logger = RunLogger(
                log_dir=log_dir,
                log_judge_calls=log_judge_calls,
                run_name=run_name,
                run_params={
                    "model": model,
                },
                cost_tracker=self.cost_tracker,
            )

        if self.verbose:
            print(f"Loaded {len(self._samples)} samples from {dataset_path}")

    def load_sample(self, sample_idx: int) -> str:
        """Load a sample by index, resetting all conversation state."""
        if not 0 <= sample_idx < len(self._samples):
            raise IndexError(
                f"Sample {sample_idx} out of range (0–{len(self._samples) - 1})"
            )

        sample = self._samples[sample_idx]
        self._sample_idx = sample_idx
        self.factual_state = sample["factual_state"]
        self._atomic_facts = sample["atomic_facts"]
        self._reset_agent()

        if self._logger is not None:
            self._logger.log_sample_loaded(
                sample_idx=sample_idx,
                original_sample_idx=sample.get("original_sample_idx"),
                factual_state=self.factual_state,
                atomic_facts=self._atomic_facts,
            )

        if self.verbose:
            print(
                f"Loaded sample {sample_idx} "
                f"({len(self.factual_state)} chars, "
                f"{len(self._atomic_facts)} atomic facts)"
            )

        return self.factual_state

    def ask(self, question: str) -> str:
        """Ask a question; the answer is collected for scoring."""
        self._require_sample()
        answer = self._agent.ask(question)
        self._all_answers.append(answer)

        if answer and answer.lower() != NO_DOCUMENT_ANSWER.lower():
            self._answers.append(answer)

        if self._logger is not None:
            self._logger.log_question(question, answer)

        if self.verbose:
            print(f"Q: {question}")
            print(f"A: {answer}\n")

        return answer

    def score(self) -> dict[str, Any]:
        """Score current discovery progress against atomic facts.

        Uses incremental evaluation: facts already confirmed as covered in
        previous calls are skipped, so only the remaining facts are sent to
        the scorer.  Only answers collected since the last scoring round are
        used as new evidence (the embedding cache still avoids redundant
        embedding calls for older answers).
        """
        self._require_sample()

        pending_indices = [
            i for i in range(len(self._atomic_facts))
            if i not in self._covered_indices
        ]
        pending_facts = [self._atomic_facts[i] for i in pending_indices]

        new_answers = self._answers[self._last_scored_answer_count:]
        self._last_scored_answer_count = len(self._answers)

        judge_details: list[dict[str, Any]] = []
        if pending_facts and new_answers:
            scorer = SemanticScorer(
                model=self.model,
                verbose=self.verbose,
                embedding_cache=self._embedding_cache,
                cost_tracker=self.cost_tracker,
            )
            partial = scorer.score(pending_facts, new_answers)
            judge_details = partial.get("judge_details", [])

            for detail in partial["facts"]:
                if detail["covered"]:
                    self._covered_indices.add(pending_indices[detail["idx"]])
        total = len(self._atomic_facts)
        covered = len(self._covered_indices)
        undiscovered = [
            self._atomic_facts[i]
            for i in range(total)
            if i not in self._covered_indices
        ]

        # Question quality: how often did the agent's questions elicit
        # useful (non-NO_DOCUMENT_ANSWER) information?
        #   1.0  — answer has no NO_DOCUMENT_ANSWER
        #   0.5  — answer contains NO_DOCUMENT_ANSWER but also other text
        #   0.0  — answer is purely NO_DOCUMENT_ANSWER
        no_doc = NO_DOCUMENT_ANSWER.lower()
        q_scores: list[float] = []
        no_doc_count = 0
        for ans in self._all_answers:
            ans_stripped = (ans or "").strip()
            ans_lower = ans_stripped.lower()
            if no_doc not in ans_lower:
                q_scores.append(1.0)
            elif ans_lower == no_doc:
                q_scores.append(0.0)
                no_doc_count += 1
            else:
                q_scores.append(0.5)
                no_doc_count += 1
        total_questions = len(self._all_answers)
        question_quality = (
            sum(q_scores) / total_questions if total_questions else 1.0
        )

        result = {
            "coverage_ratio": covered / total if total else 1.0,
            "original_facts": total,
            "covered_facts": covered,
            "undiscovered_facts": undiscovered,
            "facts": [
                {"idx": i, "text": self._atomic_facts[i],
                 "covered": i in self._covered_indices}
                for i in range(total)
            ],
            "sample_idx": self._sample_idx,
            "original_sample_idx": self._samples[self._sample_idx].get(
                "original_sample_idx"
            ),
            "questions_asked": len(self._agent.history) // 2,
            "answers_collected": len(self._answers),
            "question_quality": question_quality,
            "no_document_answers": no_doc_count,
            "total_questions": total_questions,
        }

        if self._logger is not None:
            self._logger.log_score(result, judge_details)

        return result

    def update_log_params(self, extra: dict[str, Any]) -> None:
        """Add extra key-value pairs to the run log parameters."""
        if self._logger is not None:
            self._logger.update_params(extra)

    def flush_log(self) -> Path | None:
        """Finalize and write the run log to disk. Returns file path or None."""
        if self._logger is not None:
            return self._logger.flush()
        return None

    def reset(self) -> None:
        """Reset conversation state for the current sample."""
        self._require_sample()
        self._reset_agent()

    def get_answers(self) -> list[str]:
        return list(self._answers)

    def get_history(self) -> list[dict[str, str]]:
        return list(self._agent.history) if self._agent else []

    def num_samples(self) -> int:
        return len(self._samples)

    def info(self) -> dict[str, Any]:
        return {
            "sample_idx": self._sample_idx,
            "factual_state_length": len(self.factual_state) if self.factual_state else 0,
            "model": self.model,
            "num_samples": len(self._samples),
            "questions_asked": len(self._agent.history) // 2 if self._agent else 0,
            "answers_collected": len(self._answers),
        }

    def _reset_agent(self) -> None:
        self._agent = FactChatAgent(
            document_text=self.factual_state,
            model=self.model,
            cost_tracker=self.cost_tracker,
        )
        self._answers = []
        self._all_answers = []
        self._covered_indices = set()
        self._last_scored_answer_count = 0

    def _require_sample(self) -> None:
        if self.factual_state is None:
            raise RuntimeError("No sample loaded. Call load_sample(idx) first.")

    @staticmethod
    def _load_dataset(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "No dataset ships with fsdbench by default. Pass --dataset "
                "(or dataset_path=...) pointing at a JSON file, or place "
                "raw-easy.json / raw-hard.json under data/. See data/README.md "
                "for the expected schema and how to obtain the data."
            )

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            state = (item.get("factual_state") or "").strip()
            facts = [
                s.strip()
                for s in (item.get("atomic_facts") or [])
                if s and s.strip()
            ]
            if state and facts:
                samples.append({
                    "factual_state": state,
                    "atomic_facts": facts,
                    "original_sample_idx": item.get("sample_idx"),
                })

        return samples
