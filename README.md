# fsdbench — Factual State Discovery Benchmark

Official repository for the **Factual State Discovery Benchmark: Evaluating Fact
Elicitation in Polish Tax Law** (ACL 2026 SRW).

A benchmark for evaluating how well conversational agents can **discover the
factual state** of a taxpayer's situation from Polish individual tax
interpretation documents (KIS / Eureka).

## How it works

Each sample in the dataset contains:

- **factual state** — a narrative description of the taxpayer's situation;
- **atomic facts** — a list of independent factual claims extracted from that
  narrative.

The benchmark runs a conversation between two LLM roles:

- A **QA agent** (`FactChatAgent`) role-plays the taxpayer. It answers questions
  using *only* the factual state, in first person, refusing legal advice, and
  replies `"Nie wynika z dokumentu."` when asked about something not in the
  document.
- A **discovery agent** (`DiscoveryChatbot`) plays a tax advisor conducting an
  interview, asking one question per turn to uncover all the facts.

After the conversation, the benchmark **scores** how many atomic facts were
discovered, using:

1. Embedding similarity (`text-embedding-3-small`) to find candidate matches
   between the agent's answers and the atomic facts;
2. An LLM judge that decides whether each atomic fact is fully covered by the
   matched answers.

## Quickstart (TL;DR)

```bash
# 1. Install (uv recommended; or `pip install -e .`)
uv venv && source .venv/bin/activate && uv pip install -e .

# 2. Point LiteLLM at a model provider
export OPENAI_API_KEY=sk-...                       # OpenAI directly, or...
# export OPENAI_BASE_URL=http://localhost:4000     # ...a LiteLLM proxy

# 3. Fetch the dataset from the Hugging Face Hub and write a CLI-ready JSON
python -c "import json; from datasets import load_dataset; \
json.dump(load_dataset('AI-TAX/factual-state-discovery-benchmark', split='easy').to_list(), \
open('data/raw-easy.json','w'), ensure_ascii=False)"

# 4. Run the discovery benchmark on a few samples
fsdbench run --dataset data/raw-easy.json --num_samples 3 --max_rounds 10
```

You'll see, per sample, the advisor's questions and the taxpayer's answers, a
coverage score every `--check_every` rounds, a final per-split summary, a JSON
run log in `--log_dir`, and a results file at `--output`. Example tail:

```
--- Score after 10 rounds ---
Coverage: 82.4%
Question quality: 60.0%
...
BENCHMARK SUMMARY
  Fully discovered: 1 (33.3%)
  Abandoned: 2 (66.7%)
```

> Use any model your provider/proxy serves via `--server_model` / `--chatbot_model`
> (e.g. `gpt-4o-mini`, `gpt-5-mini`, `anthropic/claude-sonnet-4-6`). The scorer
> needs an embedding model too (`text-embedding-3-small` by default).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[serve]"      # drop [serve] if you don't need the HTTP server
```

Or with plain pip:

```bash
pip install -e ".[serve]"
```

## Configuration

`fsdbench` calls LLMs and embeddings through [LiteLLM](https://docs.litellm.ai),
which reads standard provider environment variables. Copy `.env.example` and
set what you need:

```bash
export OPENAI_API_KEY=sk-...
# Optional — route through a proxy (e.g. a local LiteLLM proxy):
# export OPENAI_BASE_URL=http://localhost:4000
```

The QA agent, LLM judge, and embeddings use OpenAI models by default. The
discovery chatbot model is configurable (`--chatbot_model`) and may use any
LiteLLM-supported provider (e.g. `anthropic/claude-sonnet-4-6`).

## Data

The dataset is published on the Hugging Face Hub:
**[`AI-TAX/factual-state-discovery-benchmark`](https://huggingface.co/datasets/AI-TAX/factual-state-discovery-benchmark)**
(splits `easy`: 250 samples / 7,132 facts and `hard`: 250 / 25,742; 32,874
atomic facts total). No data ships in this repo. The dataset is licensed
**CC BY 4.0** (this repo's code is MIT).

```python
from datasets import load_dataset

ds = load_dataset("AI-TAX/factual-state-discovery-benchmark")
easy, hard = ds["easy"], ds["hard"]
```

The `fsdbench run` CLI loads a local JSON file (a list of `{factual_state,
atomic_facts}` objects). Materialise a split into that form once:

```python
import json
from datasets import load_dataset
for split in ("easy", "hard"):
    rows = load_dataset("AI-TAX/factual-state-discovery-benchmark", split=split).to_list()
    json.dump(rows, open(f"data/raw-{split}.json", "w"), ensure_ascii=False)
```

See [`data/README.md`](data/README.md) for the schema.

## Quick start

### Run the benchmark (discovery agent vs QA agent)

```bash
# 10 easy samples, gpt-4o-mini for both roles, up to 30 rounds each
fsdbench run \
  --dataset data/raw-easy.json \
  --num_samples 10 --max_rounds 30 --check_every 5 \
  --server_model gpt-4o-mini --chatbot_model gpt-4o-mini \
  --log_dir logs --output discovery_results.json
```

Per-sample run logs (full Q&A + score checkpoints) are written to `--log_dir`;
a results summary is written to `--output`. Re-running with the same parameters
restores already-completed samples from prior logs (disable with `--no_restore`).

To reproduce the paper-scale runs (25 parallel processes per split):

```bash
bash scripts/run_easy.sh   # baseline prompt; pass "main" for the structured prompt
bash scripts/run_hard.sh
```

### Other commands

```bash
fsdbench demo --dataset data/raw-easy.json --sample 0   # 3 example questions, then score
fsdbench interactive --dataset data/raw-easy.json       # manual Q&A session
fsdbench serve --dataset data/raw-easy.json --port 8000 # HTTP server
```

### Python API

```python
from fsdbench import BenchmarkServer

server = BenchmarkServer(dataset_path="data/raw-easy.json")
server.load_sample(0)

server.ask("Kim jest wnioskodawca?")
server.ask("Jaką działalność prowadzi?")

result = server.score()
print(f"Coverage: {result['coverage_ratio']:.1%}")
```

To drive it with the discovery agent programmatically:

```python
from fsdbench import BenchmarkServer, DiscoveryChatbot

server = BenchmarkServer(dataset_path="data/raw-easy.json", model="gpt-4o-mini")
chatbot = DiscoveryChatbot(model="gpt-4o-mini", max_rounds=30, cost_tracker=server.cost_tracker)
result = chatbot.discover_sample(server, sample_idx=0)
print(result["status"], result["coverage"])
```

## HTTP API

| Method | Endpoint        | Description                      |
| ------ | --------------- | -------------------------------- |
| POST   | `/load_sample`  | Load a sample by index           |
| POST   | `/ask`          | Ask a question                   |
| GET    | `/score`        | Score current discovery progress |
| POST   | `/reset`        | Reset conversation state         |
| GET    | `/answers`      | List collected answers           |
| GET    | `/history`      | Full conversation history        |
| GET    | `/info`         | Server status                    |

## Scoring output

`server.score()` returns:

| Key                  | Type        | Description                                  |
| -------------------- | ----------- | -------------------------------------------- |
| `coverage_ratio`     | `float`     | Fraction of atomic facts covered (0–1)       |
| `original_facts`     | `int`       | Total atomic facts in the sample             |
| `covered_facts`      | `int`       | Number of facts judged as covered            |
| `undiscovered_facts` | `list[str]` | Atomic facts not yet discovered              |
| `questions_asked`    | `int`       | Total questions asked so far                 |
| `question_quality`   | `float`     | Fraction of questions that elicited info     |
| `facts`              | `list`      | Per-fact detail (idx, text, covered)         |

## Results (ACL 2026 SRW)

Headline results from the paper. Four models act as the **discovery agent**,
running 50 turns of dialogue per sample on the easy and hard splits.
$C(\pi)$ = mean Fact Coverage Ratio (%) ± std; all models used the full
$Q(\pi)=50$-turn budget. Best per split in **bold**.

| Model | Easy $C(\pi)$ | Hard $C(\pi)$ | Mean $C$ |
| ----- | :-----------: | :-----------: | :------: |
| GPT-OSS-120B *(open)* | 74.1 ± 20.0 | 34.3 ± 17.2 | 54.2 |
| DeepSeek V3-0324 *(open)* | 70.7 ± 20.5 | 38.4 ± 17.1 | 54.5 |
| GPT-5 Mini *(closed)* | 69.8 ± 22.1 | 29.2 ± 13.5 | 49.5 |
| Claude Sonnet 4.6 *(closed)* | **77.2 ± 17.3** | **48.6 ± 15.7** | **62.9** |

Even the best system recovers only **77% of facts on easy** samples and **under
49% on hard** samples — conversational fact elicitation remains an open problem.
Coverage roughly halves from the easy to the hard split, and instance-level
variance is high (13.5–22.1 pp std).

**Prompt ablation** (50 samples each; $\Delta$ = domain-expert − baseline prompt):
a minimal baseline prompt generally beats a domain-structured ORD-IN prompt —
DeepSeek V3 drops −15.2 pp on easy and −5.0 pp on hard; GPT-5 Mini −3.0 pp on
easy but **+3.9 pp on hard**, suggesting structure helps only when the baseline
strategy struggles on longer documents.

**Dataset & ground truth.** 500 narratives from official Polish tax
interpretations → 32,874 atomic facts (easy: 250 samples / 7,132 facts; hard:
250 / 25,742), published at
[`AI-TAX/factual-state-discovery-benchmark`](https://huggingface.co/datasets/AI-TAX/factual-state-discovery-benchmark).
Atomic-fact extraction quality (50 docs): supported precision 97.6%, atomicity
93.8%, sentence coverage 96.0%.

## Citation

```bibtex
@inproceedings{bystronski2026fsdbench,
  title     = {Factual State Discovery Benchmark: Evaluating Fact Elicitation in Polish Tax Law},
  author    = {Bystro\'nski, Mateusz and Tagowski, Kamil and Janiak, Denis and
               Farganus, Julia and Augustyniak, {\L}ukasz and
               Kajdanowicz, Monika and Kajdanowicz, Tomasz},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational
               Linguistics: Student Research Workshop (ACL SRW)},
  year      = {2026},
  address   = {Wroclaw University of Science and Technology}
}
```

## License

MIT — see [LICENSE](LICENSE).
