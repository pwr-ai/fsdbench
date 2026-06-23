from __future__ import annotations

import argparse

from . import discovery
from .server import BenchmarkServer


# ---------------------------------------------------------------------------
# Server construction shared by demo / interactive
# ---------------------------------------------------------------------------

def _build_server(args, *, verbose: bool) -> BenchmarkServer:
    kwargs = dict(
        model=args.model,
        verbose=verbose,
        log_dir=args.log_dir,
        log_judge_calls=args.log_judge,
        run_name=args.run_name,
    )
    if args.dataset:
        kwargs["dataset_path"] = args.dataset
    return BenchmarkServer(**kwargs)


# ---------------------------------------------------------------------------
# Subcommand: interactive
# ---------------------------------------------------------------------------

def interactive(args) -> None:
    """Run an interactive CLI session."""
    server = _build_server(args, verbose=True)
    n = server.num_samples()

    print(f"\nFactual State Discovery Benchmark ({n} samples)")
    print("Commands: /load <idx>, /score, /answers, /reset, /show, /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            log_path = server.flush_log()
            if log_path:
                print(f"\nRun log saved to {log_path}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            log_path = server.flush_log()
            if log_path:
                print(f"Run log saved to {log_path}")
            break
        elif user_input.startswith("/load "):
            _cmd_load(server, user_input)
        elif user_input == "/score":
            _cmd_score(server)
        elif user_input == "/answers":
            _cmd_answers(server)
        elif user_input == "/reset":
            _cmd_reset(server)
        elif user_input == "/show":
            _cmd_show(server)
        else:
            if server.factual_state is None:
                print("No sample loaded. Use /load <idx>\n")
                continue
            server.ask(user_input)

    print("\nGoodbye!")


# ---------------------------------------------------------------------------
# Subcommand: demo
# ---------------------------------------------------------------------------

def demo(args) -> None:
    """Run a short demo with example questions."""
    server = _build_server(args, verbose=True)

    if server.num_samples() == 0:
        print("No valid samples found in the dataset.")
        return

    idx = args.sample if 0 <= args.sample < server.num_samples() else 0
    server.load_sample(idx)

    print(f"\nFactual state ({len(server.factual_state)} chars):")
    print(f"{server.factual_state[:300]}...\n")

    for question in [
        "Kim jest wnioskodawca?",
        "Jaką działalność prowadzi?",
        "Od kiedy prowadzi działalność?",
    ]:
        server.ask(question)

    result = server.score()
    print(f"\nCoverage: {result['coverage_ratio']:.1%}")

    log_path = server.flush_log()
    if log_path:
        print(f"Run log saved to {log_path}")


# ---------------------------------------------------------------------------
# Subcommand: serve
# ---------------------------------------------------------------------------

def serve(args) -> None:
    """Start the HTTP server."""
    from .api import serve as serve_app

    serve_app(
        model=args.model,
        port=args.port,
        dataset_path=args.dataset,
        log_dir=args.log_dir,
        log_judge_calls=args.log_judge,
        run_name=args.run_name,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt-4o-mini", help="Model for QA agent / judge / embeddings")
    p.add_argument("--dataset", default=None, help="Path to dataset JSON (default: data/raw-easy.json)")
    p.add_argument("--log-dir", dest="log_dir", default=None,
                   help="Directory for run logs (enables logging when set)")
    p.add_argument("--log-judge", dest="log_judge", action="store_true",
                   help="Include raw judge LLM calls/responses in run logs")
    p.add_argument("--run-name", dest="run_name", default=None,
                   help="Human-readable run name (used in log filename and JSON)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fsdbench",
        description="Factual State Discovery Benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run --------------------------------------------------------------------
    run_p = sub.add_parser("run", help="Run the discovery-agent benchmark")
    run_p.add_argument("--start_sample", type=int, default=0,
                       help="Starting sample index (default: 0)")
    run_p.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to process (default: 5)")
    run_p.add_argument("--max_rounds", type=int, default=30,
                       help="Max rounds per sample before abandoning (default: 30)")
    run_p.add_argument("--check_every", type=int, default=5,
                       help="Check score every N rounds (default: 5)")
    run_p.add_argument("--server_model", type=str, default="gpt-4o-mini",
                       help="Model for benchmark server (QA agent, judge, embeddings) (default: gpt-4o-mini)")
    run_p.add_argument("--chatbot_model", type=str, default=None,
                       help="Model for the discovery chatbot (e.g. clarin/or-gemma-3-27b-it). "
                            "Defaults to --server_model if not set.")
    run_p.add_argument("--output", type=str, default="discovery_results.json",
                       help="Output file for results (default: discovery_results.json)")
    run_p.add_argument("--prompt", type=str, default="main",
                       choices=list(discovery.DISCOVERY_PROMPTS.keys()),
                       help="Discovery prompt variant (default: main)")
    run_p.add_argument("--no_gaps_hint", action="store_true",
                       help="Disable showing undiscovered text to guide questions")
    run_p.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    run_p.add_argument("--no_restore", action="store_true",
                       help="Disable automatic restore of already-computed samples from previous runs")
    run_p.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset JSON file (default: data/raw-easy.json)")
    run_p.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for run logs (default: logs)")
    run_p.set_defaults(func=discovery.run)

    # demo -------------------------------------------------------------------
    demo_p = sub.add_parser("demo", help="Ask 3 example questions and score one sample")
    _add_server_args(demo_p)
    demo_p.add_argument("--sample", type=int, default=0, help="Demo sample index")
    demo_p.set_defaults(func=demo)

    # interactive ------------------------------------------------------------
    inter_p = sub.add_parser("interactive", help="Interactive CLI session")
    _add_server_args(inter_p)
    inter_p.set_defaults(func=interactive)

    # serve ------------------------------------------------------------------
    serve_p = sub.add_parser("serve", help="Run the HTTP server")
    _add_server_args(serve_p)
    serve_p.add_argument("--port", type=int, default=8000, help="HTTP server port")
    serve_p.set_defaults(func=serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _cmd_load(server: BenchmarkServer, user_input: str) -> None:
    try:
        idx = int(user_input.split()[1])
        server.load_sample(idx)
        print(f"Loaded sample {idx} ({len(server.factual_state)} chars)\n")
    except (IndexError, ValueError) as e:
        print(f"Error: {e}\n")


def _cmd_score(server: BenchmarkServer) -> None:
    if server.factual_state is None:
        print("No sample loaded. Use /load <idx>\n")
        return

    result = server.score()
    print(f"\n=== SCORE (Sample {result['sample_idx']}) ===")
    print(f"Questions asked: {result['questions_asked']}")
    print(f"Answers collected: {result['answers_collected']}")
    print(
        f"Coverage: {result['coverage_ratio']:.1%} "
        f"({result['covered_facts']}/{result['original_facts']} facts)"
    )
    if result["undiscovered_facts"]:
        print("\nUndiscovered facts (first 5):")
        for fact in result["undiscovered_facts"][:5]:
            display = fact[:80] + "..." if len(fact) > 80 else fact
            print(f"  - {display}")
    print()


def _cmd_answers(server: BenchmarkServer) -> None:
    answers = server.get_answers()
    print(f"\n=== COLLECTED ANSWERS ({len(answers)}) ===")
    for i, ans in enumerate(answers, 1):
        display = ans[:150] + "..." if len(ans) > 150 else ans
        print(f"{i}. {display}")
    print()


def _cmd_reset(server: BenchmarkServer) -> None:
    if server.factual_state is None:
        print("No sample loaded. Use /load <idx>\n")
        return
    server.reset()
    print("Reset.\n")


def _cmd_show(server: BenchmarkServer) -> None:
    if server.factual_state is None:
        print("No sample loaded. Use /load <idx>\n")
        return
    print(f"\n=== FACTUAL STATE ===\n{server.factual_state}\n")
