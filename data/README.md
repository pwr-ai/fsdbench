# Datasets

The benchmark data is **published on the Hugging Face Hub** and is **not
committed** to this repository:

**[`AI-TAX/factual-state-discovery-benchmark`](https://huggingface.co/datasets/AI-TAX/factual-state-discovery-benchmark)**
 — license **CC BY 4.0** (the code in this repo is MIT).

| File (local) | Split | Samples | Atomic facts |
| ------------ | ----- | ------- | ------------ |
| `raw-easy.json` | easy | 250 | 7,132 |
| `raw-hard.json` | hard | 250 | 25,742 |
| **Total** | | **500** | **32,874** |

`raw-easy.json` is the default the CLI looks for (`fsdbench/server.py` →
`data/raw-easy.json`). Point at any file with `--dataset <path>`.

## How to obtain the data

Pull the splits from the Hub and write them as the JSON files the CLI expects
(a JSON array of sample objects):

```python
import json
from datasets import load_dataset

for split in ("easy", "hard"):
    rows = load_dataset("AI-TAX/factual-state-discovery-benchmark", split=split).to_list()
    json.dump(rows, open(f"data/raw-{split}.json", "w"), ensure_ascii=False)
```

Then:

```bash
fsdbench run --dataset data/raw-easy.json --num_samples 10
```

## Schema

Each local JSON file is an array of sample objects. Only two fields are
required; any extra fields are ignored by the loader
(`BenchmarkServer._load_dataset`).

```json
[
  {
    "factual_state": "W przedmiotowym wniosku ...",   // required: narrative of the taxpayer's situation
    "atomic_facts": [                                   // required: independent factual claims
      "Wnioskodawczyni jest osobą fizyczną.",
      "Otrzymała środki z tytułu wygaśnięcia prawa do lokalu.",
      "..."
    ],
    "sample_idx": 410249,                               // optional: original source index (kept as original_sample_idx)
    "id": 123762,                                       // optional: source document id
    "factual_state_length": 998,                        // optional, ignored
    "atomic_facts_count": 17                            // optional, ignored
  }
]
```

Samples with an empty `factual_state` or an empty `atomic_facts` list are
dropped at load time.
