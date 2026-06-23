#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discovery Chatbot - an LLM agent that discovers factual state through conversation.

The chatbot:
1. Asks questions to discover facts from the factual state.
2. Checks the score every N rounds.
3. Moves to the next sample once 100% coverage is reached.
4. Abandons a sample once the round budget is exhausted without full coverage.

Run via the CLI:
    fsdbench run --dataset data/raw-easy.json --num_samples 10
"""

import argparse
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path

import litellm

from .event_logger import CostTracker
from .server import BenchmarkServer

# Map custom provider prefixes to OpenAI-compatible endpoints.
# Each entry: prefix -> (api_base_env, api_key_env)
_CUSTOM_PROVIDERS = {
    "clarin/": ("CLARIN_API_BASE", "CLARIN_API_KEY"),
}


def llm_completion(*, model: str, messages, **kwargs):
    """litellm.completion wrapper that routes custom provider prefixes."""
    for prefix, (base_env, key_env) in _CUSTOM_PROVIDERS.items():
        if model.startswith(prefix):
            kwargs["api_base"] = os.environ[base_env]
            kwargs["api_key"] = os.environ[key_env]
            model = "openai/" + model[len(prefix):]
            break
    return litellm.completion(model=model, messages=messages, **kwargs)


# ============================================================================
# DISCOVERY CHATBOT
# ============================================================================

# =============================================================================
# PROMPT: DISCOVERY AGENT — BENCHMARK VERSION
# =============================================================================
# Cel: zbieranie faktów od klienta (symulowanego) w celu przygotowania
# wniosku o indywidualną interpretację podatkową (ORD-IN).
#
# Wersje:
#   DISCOVERY_BASELINE  — naiwna (ablation: brak struktury domenowej)
#   DISCOVERY_MAIN      — szczegółowa (main system)
# =============================================================================


DISCOVERY_BASELINE = """\
Jesteś doradcą podatkowym. Klient chce złożyć wniosek o indywidualną \
interpretację podatkową (ORD-IN). Twoim zadaniem jest zebrać od niego \
wszystkie informacje potrzebne do przygotowania tego wniosku.

Zadawaj po jednym pytaniu na raz. Pytaj w języku polskim.
"""


DISCOVERY_MAIN = """\
# ROLA

Jesteś doświadczonym doradcą podatkowym, który przeprowadza wywiad z klientem w celu zebrania kluczowych informacji mających na celu ustalenie stanu faktycznego do wniosku o indywidualną interpretację podatkową (ORD-IN) składanego do Dyrektora Krajowej Informacji Skarbowej.

# CEL

Zebrać minimalny, ale kompletny i wyczerpujący zestaw faktów oraz informacji niezbędnych do sporządzenia trzech elementów wniosku ORD-IN:

1. Opis stanu faktycznego (lub zdarzenia przyszłego)
2. Pytanie podatkowe
3. Własne stanowisko wnioskodawcy w sprawie oceny prawnej stanu faktycznego (lub zdarzenia przyszłego)

# ZAKRES INFORMACJI DO ZEBRANIA

Poniższe kategorie wyznaczają zakres wywiadu. Zbieraj informacje w podanej kolejności, pomijając punkty, które nie mają wpływu na kwalifikację podatkową opisywanego problemu.

## A. Wnioskodawca

- Rodzaj wnioskodawcy: 1) osoba fizyczna / 2) osoba prawna / 3) jednostka organizacyjna niemająca osobowości prawnej / 4) podatkowa grupa kapitałowa / 5) inny
  - Doszczegółowienie: 1a) osoba fizyczna nieprowadząca działalności gosp. / 1b) osoba fizyczna prowadząca działalność gosp. / 1c) wspólnik spółki jawnej / 1d) wspólnik spółki cywilnej / 1e) wspólnik spółki partnerskiej / 1f) inny; 2a) sp. z o.o. / 2b) SA / 2c) spółka komandytowa / 2d) spółka komandytowo-akcyjna / 2e) inna
- Status VAT: czynny podatnik / korzystający ze zwolnienia / niezarejestrowany dla celów VAT / podatek nie dotyczy sprawy
- Forma opodatkowania dochodów: skala podatkowa / podatek liniowy 19% / ryczałt od przychodów ewidencjonowanych / karta podatkowa / CIT / inna

## B. Istota problemu podatkowego

- Konkretna transakcja, czynność lub zdarzenie budzące wątpliwość
- Podatek, którego dotyczy wątpliwość: PIT / CIT / VAT / PCC / akcyza / podatek od nieruchomości / podatek od spadków i darowizn / Ordynacja podatkowa / podatek od wydobycia niektórych kopalin / inne (np. podatek od gier, podatek od niektórych instytucji finansowych, zasady ewidencji i identyfikacji podatników i płatników) – wg ORD-IN
- Konkretny przepis podatkowy (art. ust. pkt konkretnej ustawy), o interpretację którego występuje podatnik – jest to element konieczny i kluczowy wniosku, do niego odnosi się zapytanie wnioskodawcy (bezpośrednio lub pośrednio), a organ podatkowy dokonuje interpretacji / wykładni tego konkretnego przepisu
- Charakter czasowy: stan faktyczny już zaistniały czy zdarzenie przyszłe (planowane)

## C. Fakty transakcji (tylko istotne dla kwalifikacji podatkowej, przedstawione w sposób jednoznaczny i wyczerpujący)

- Przedmiot: towar / usługa / prawo majątkowe / instrument finansowy / nieruchomość / inne
- Strony i ich relacje: czy występują powiązania w rozumieniu art. 11a CIT lub art. 23m PIT; kraj siedziby/zamieszkania kontrahenta
- Podstawa prawna czynności: rodzaj umowy, kluczowe postanowienia umowne wpływające na kwalifikację podatkową
- Sposób rozliczenia: waluta (tylko jeśli ma znaczenie dla przedmiotu zapytania), sposób dokumentowania (faktura / rachunek / inna forma), terminy i forma płatności
- Kontekst regulacyjny: czy transakcja podlega szczególnym reżimom (np. odwrotne obciążenie, zwolnienie przedmiotowe, ulga, procedura szczególna VAT, ceny transferowe)

## D. Stanowisko wnioskodawcy

- Jak klient uważa, że powinien potraktować dla celów podatkowych / rozliczyć dla celów podatkowych opisaną transakcję / zdarzenie
- Na jakiej podstawie prawnej opiera swoje stanowisko (przepis ustawy, rozporządzenia, wcześniejsza interpretacja – jeśli zna)
- Jeśli klient nie ma własnego stanowiska, zanotuj ten fakt i przejdź dalej

# ZASADY PROWADZENIA WYWIADU

1. Zadawaj JEDNO pytanie na turę.
2. Pytaj WYŁĄCZNIE o elementy z zakresu A–D, których jeszcze nie znasz.
3. Pomijaj pytania, których odpowiedź nie wpłynie na kwalifikację podatkową opisywanego problemu.
4. Jeśli odpowiedź klienta ujawnia dodatkowy wątek podatkowy (np. kwestia VAT i jednocześnie PIT), zbierz fakty dla obu wątków.
5. Jeśli klient odpowiada „nie wiem" lub „nie dotyczy", zaakceptuj odpowiedź i przejdź do następnego pytania.
6. Nie udzielaj porad prawnych, nie interpretuj przepisów, nie oceniaj stanowiska klienta – Twoim jedynym zadaniem jest zbieranie faktów oraz niezbędnych informacji.

# FORMAT ODPOWIEDZI

Odpowiadaj WYŁĄCZNIE treścią jednego pytania w języku polskim. Bez numeracji, bez wstępów, bez komentarzy, bez podsumowań.
"""

DISCOVERY_PROMPTS = {
    "baseline": DISCOVERY_BASELINE,
    "main": DISCOVERY_MAIN,
}


@dataclass
class DiscoveryChatbot:
    """
    LLM-based chatbot that discovers factual state through conversation.
    """

    model: str = "gpt-4o-mini"
    max_rounds: int = 30
    check_score_every: int = 5
    verbose: bool = True
    use_gaps_hint: bool = True
    prompt_name: str = "main"
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    _conversation_history: List[Dict[str, str]] = field(default_factory=list, repr=False)

    def _generate_question(self, undiscovered_text: str | None = None) -> str:
        """Generate next question based on conversation history."""
        base_prompt = DISCOVERY_PROMPTS.get(self.prompt_name, DISCOVERY_MAIN)
        if self.use_gaps_hint and undiscovered_text:
            system_prompt = (
                base_prompt.rstrip()
                + "\n\nNIEODKRYTE INFORMACJE (musisz o nie zapytać):\n"
                + undiscovered_text[:1000]
            )
        else:
            system_prompt = base_prompt

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history
        if self._conversation_history:
            history_text = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in self._conversation_history
            ])
            messages.append({
                "role": "user",
                "content": f"Dotychczasowa rozmowa:\n{history_text}\n\nZadaj następne pytanie:"
            })
        else:
            messages.append({
                "role": "user",
                "content": "Rozpocznij rozmowę. Zadaj pierwsze pytanie, aby poznać podstawowe fakty o kliencie:"
            })

        resp = llm_completion(
            model=self.model,
            messages=messages,
        )
        self.cost_tracker.track("discovery", resp)

        question = (resp.choices[0].message.content or "").strip()
        return question

    def discover_sample(
        self,
        server: BenchmarkServer,
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        Discover factual state for a single sample.

        Returns:
            Dict with discovery results
        """
        # Load sample
        server.load_sample(sample_idx)
        self._conversation_history = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Sample {sample_idx} ({len(server.factual_state)} chars)")
            print(f"{'='*60}")

        round_num = 0
        last_score = None
        undiscovered_text = None

        while round_num < self.max_rounds:
            round_num += 1

            # Generate question
            question = self._generate_question(undiscovered_text)

            if self.verbose:
                print(f"\n[Round {round_num}] Q: {question}")

            # Ask the server
            answer = server.ask(question)

            if self.verbose:
                print(f"           A: {answer}")

            # Record conversation
            self._conversation_history.append({
                "question": question,
                "answer": answer,
            })

            # Check score every N rounds
            if round_num % self.check_score_every == 0:
                score_result = server.score()
                coverage = score_result["coverage_ratio"]
                undiscovered_text = " ".join(score_result.get("undiscovered_facts", []))

                question_quality = score_result.get("question_quality", 1.0)
                no_doc_count = score_result.get("no_document_answers", 0)

                if self.verbose:
                    print(f"\n--- Score after {round_num} rounds ---")
                    print(f"Coverage: {coverage:.1%}")
                    print(f"Question quality: {question_quality:.1%} ({no_doc_count} no-doc answers)")

                last_score = score_result

                # Check if fully discovered
                if coverage >= 1.0:
                    if self.verbose:
                        print(f"\n✓ FULLY DISCOVERED in {round_num} rounds!")

                    return {
                        "sample_idx": sample_idx,
                        "status": "discovered",
                        "rounds": round_num,
                        "coverage": coverage,
                        "question_quality": question_quality,
                        "no_document_answers": no_doc_count,
                        "questions_asked": len(self._conversation_history),
                        "conversation": self._conversation_history,
                    }

        # Abandoned after max rounds
        final_score = server.score()
        final_quality = final_score.get("question_quality", 1.0)
        final_no_doc = final_score.get("no_document_answers", 0)

        if self.verbose:
            print(f"\n✗ ABANDONED after {round_num} rounds")
            print(f"Final coverage: {final_score['coverage_ratio']:.1%}")
            print(f"Question quality: {final_quality:.1%} ({final_no_doc} no-doc answers)")

        return {
            "sample_idx": sample_idx,
            "status": "abandoned",
            "rounds": round_num,
            "coverage": final_score["coverage_ratio"],
            "question_quality": final_quality,
            "no_document_answers": final_no_doc,
            "questions_asked": len(self._conversation_history),
            "conversation": self._conversation_history,
        }

    def run_benchmark(
        self,
        server: BenchmarkServer,
        sample_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Run discovery on multiple samples.

        Args:
            server: BenchmarkServer instance
            sample_indices: List of sample indices to process

        Returns:
            List of results for each sample
        """
        results = []

        for idx in sample_indices:
            try:
                result = self.discover_sample(server, idx)
                results.append(result)
            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                results.append({
                    "sample_idx": idx,
                    "status": "error",
                    "error": str(e),
                })

        return results


def print_summary(
    results: List[Dict[str, Any]],
    cost_tracker: CostTracker | None = None,
) -> None:
    """Print summary of benchmark results."""
    if not results:
        print("No results.")
        return

    discovered = [r for r in results if r.get("status") == "discovered"]
    abandoned = [r for r in results if r.get("status") == "abandoned"]
    errors = [r for r in results if r.get("status") == "error"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(results)}")
    print(f"  Fully discovered: {len(discovered)} ({len(discovered)/len(results):.1%})")
    print(f"  Abandoned: {len(abandoned)} ({len(abandoned)/len(results):.1%})")
    print(f"  Errors: {len(errors)}")

    if discovered:
        avg_rounds = sum(r["rounds"] for r in discovered) / len(discovered)
        print(f"\nDiscovered samples:")
        print(f"  Avg rounds to discover: {avg_rounds:.1f}")

    if abandoned:
        avg_coverage = sum(r["coverage"] for r in abandoned) / len(abandoned)
        print(f"\nAbandoned samples:")
        print(f"  Avg coverage: {avg_coverage:.1%}")

    # Question quality across all non-error results
    scored = [r for r in results if r.get("question_quality") is not None]
    if scored:
        avg_quality = sum(r["question_quality"] for r in scored) / len(scored)
        total_no_doc = sum(r.get("no_document_answers", 0) for r in scored)
        total_qs = sum(r.get("questions_asked", 0) for r in scored)
        print(f"\nQuestion quality:")
        print(f"  Avg quality: {avg_quality:.1%}")
        print(f"  No-document answers: {total_no_doc}/{total_qs}")

    if cost_tracker is not None and cost_tracker.total_calls > 0:
        summary = cost_tracker.summary()
        print(f"\n--- COSTS ---")
        print(f"  Total: ${summary['total_cost_usd']:.6f}")
        print(f"  LLM calls: {summary['total_calls']}")
        for cat, data in summary["by_category"].items():
            tokens = data["total_tokens"]
            print(
                f"  {cat}: {data['calls']} calls, "
                f"${data['cost_usd']:.6f}, "
                f"{tokens} tokens"
            )


_RESTORE_MATCH_KEYS = ("model", "max_rounds", "check_score_every", "use_gaps_hint", "chatbot_model", "prompt")


def _count_qa(sample: Dict[str, Any]) -> int:
    """Number of question events (one Q&A pair per event)."""
    events = sample.get("events", [])
    return sum(1 for e in events if e.get("type") == "question")


def _is_solved(sample: Dict[str, Any]) -> bool:
    """True if the sample reached coverage_ratio == 1.0 (fully discovered)."""
    for e in sample.get("events", []):
        if e.get("type") == "score":
            if e.get("result", {}).get("coverage_ratio", 0) >= 1.0:
                return True
    return False


def find_restorable_samples(
    log_dir: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Scan *log_dir* for runs whose parameters match *params* on the keys
    that define the experiment (ignoring ``start_sample`` and ``num_samples``).
    Loads only run_*.json (never .bak). Restores only complete samples:
    either max_rounds Q&A pairs or coverage_ratio == 1.0.

    Returns a deduplicated list of completed sample dicts (keyed by
    ``sample_idx``, latest run wins).
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []

    match_values = {k: params.get(k) for k in _RESTORE_MATCH_KEYS}
    expected_qa = params.get("max_rounds", 50)
    restored: Dict[int, Dict[str, Any]] = {}

    for fpath in sorted(log_path.glob("run_*.json")):
        if ".bak" in fpath.name:
            continue
        try:
            with fpath.open(encoding="utf-8") as f:
                run = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        run_params = run.get("parameters", {})
        if not all(run_params.get(k) == v for k, v in match_values.items()):
            continue
        run_expected = run_params.get("max_rounds", expected_qa)

        # Only trust samples that fall within the run's own requested range
        # (guards against corrupted logs from the old restore_samples bug).
        run_start = run_params.get("start_sample")
        run_num = run_params.get("num_samples")
        if run_start is not None and run_num is not None:
            run_valid = set(range(run_start, run_start + run_num))
        else:
            run_valid = None

        for sample in run.get("samples", []):
            idx = sample.get("sample_idx")
            if idx is None:
                continue
            if run_valid is not None and idx not in run_valid:
                continue
            if _count_qa(sample) != run_expected and not _is_solved(sample):
                continue
            restored[idx] = sample

    return list(restored.values())


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSON file."""
    # Remove conversation for cleaner output (can be large)
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "conversation"}
        clean_results.append(clean)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def run(args: argparse.Namespace) -> None:
    """Execute a discovery benchmark run from parsed CLI arguments."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        mask = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
        print(f"OPENAI_API_KEY (litellm): {mask}")
    else:
        print("OPENAI_API_KEY (litellm): not set")

    chatbot_model = args.chatbot_model or args.server_model

    run_name = f"{chatbot_model}_n{args.num_samples}_r{args.max_rounds}"
    print("Initializing benchmark server...")
    server_kwargs = dict(
        model=args.server_model, verbose=False,
        log_dir=args.log_dir, log_judge_calls=True,
        run_name=run_name,
    )
    if args.dataset:
        server_kwargs["dataset_path"] = args.dataset
    server = BenchmarkServer(**server_kwargs)

    chatbot = DiscoveryChatbot(
        model=chatbot_model,
        max_rounds=args.max_rounds,
        check_score_every=args.check_every,
        verbose=not args.quiet,
        use_gaps_hint=not args.no_gaps_hint,
        prompt_name=args.prompt,
        cost_tracker=server.cost_tracker,
    )

    server.update_log_params({
        "start_sample": args.start_sample,
        "num_samples": args.num_samples,
        "max_rounds": args.max_rounds,
        "check_score_every": args.check_every,
        "use_gaps_hint": not args.no_gaps_hint,
        "chatbot_model": chatbot_model,
        "prompt": args.prompt,
    })

    # Determine sample indices
    max_samples = server.num_samples()
    end_sample = min(args.start_sample + args.num_samples, max_samples)
    sample_indices = list(range(args.start_sample, end_sample))

    # Restore already-computed samples from previous matching runs
    restored_results: List[Dict[str, Any]] = []
    if not args.no_restore:
        restore_params = {
            "model": args.server_model,
            "max_rounds": args.max_rounds,
            "check_score_every": args.check_every,
            "use_gaps_hint": not args.no_gaps_hint,
            "chatbot_model": chatbot_model,
            "prompt": args.prompt,
        }
        restored_samples = find_restorable_samples(args.log_dir, restore_params)
        if restored_samples:
            requested = set(range(args.start_sample, end_sample))
            restored_idx = {s["sample_idx"] for s in restored_samples}
            before = len(sample_indices)
            sample_indices = [i for i in sample_indices if i not in restored_idx]
            n_skipped = before - len(sample_indices)
            if n_skipped:
                print(f"Restored {n_skipped} samples from previous runs, "
                      f"{len(sample_indices)} remaining")

                if server._logger is not None:
                    relevant = [s for s in restored_samples
                                if s["sample_idx"] in requested]
                    server._logger.restore_samples(relevant)

    print(f"Running discovery on samples {args.start_sample} to {end_sample - 1} "
          f"({len(sample_indices)} to compute)")
    print(f"Max rounds per sample: {args.max_rounds}")
    print(f"Check score every: {args.check_every} rounds")

    # Run benchmark
    results = chatbot.run_benchmark(server, sample_indices)

    # Flush run log
    log_path = server.flush_log()
    if log_path:
        print(f"\nRun log saved to {log_path}")

    # Print summary
    print_summary(results, server.cost_tracker)

    # Save results
    save_results(results, args.output)
