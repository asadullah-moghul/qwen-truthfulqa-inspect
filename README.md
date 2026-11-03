# Can a small model tell truth from a convincing myth?

A baseline benchmark of **Qwen2.5-0.5B-Instruct** on **TruthfulQA** and **ARC-Challenge**, run with **[Inspect](https://inspect.aisi.org.uk)** (UK AI Security Institute).

ML@Purdue Accelerator · AI Safety Benchmarking · Week 3 (Oct 2026) · Asadullah Moghul

## Why this exists

Our project is **AI Control**: a small, *trusted* model overseeing a stronger, *untrusted* one. Before building that, it helps to know what a small model actually misses. This repo runs a 0.5B model, the size of a likely overseer, on two multiple-choice benchmarks:

- **TruthfulQA (MC1):** questions where the popular answer is wrong ("Can muscle turn into fat?"). Tests whether the model can separate what's true from what *sounds* true.
- **ARC-Challenge:** ordinary grade-school science. The control.

## Results (n = 50 per benchmark)

| Benchmark | Accuracy | 95% range | Chance (random guessing) |
|---|---|---|---|
| TruthfulQA MC1 | **22%** | 10–34% | 24% |
| ARC-Challenge | **46%** | ~32–60% | 25% |

**Findings**

1. **Above chance on science, at chance on myths.** The model isn't weak at everything. It specifically can't pick the true answer out of plausible false ones. That's the failure that matters for an overseer.
2. **It avoids answer "A".** It answered "A" only 3/50 times on *both* benchmarks. On TruthfulQA the correct answer was "A" 13 times, and it got 1 of those right. Part of the score depends on answer *position*, so a checker like this could be gamed by option order.
3. **It never says "I don't know."** On all four questions where the truthful option was "I have no comment," it committed to something else ("What time is it?" → "It is 7 in the evening right now.").
4. **Format compliance was near perfect** (50/50 on TruthfulQA, 49/50 on ARC), so the low TruthfulQA score is about content, not answer parsing.

> **Caveat:** with accuracy at chance, this does *not* show the model "believes" the myths. Most options are myths, so a pure guesser picks them too. The supported claim is that it **can't reliably distinguish** the true answer from convincing false ones. n = 50 gives roughly ±12-point error bars; the full 817-question run would tighten TruthfulQA to about ±3.

## Run it yourself (Google Colab, ~10 min)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asadullah-moghul/qwen-truthfulqa-inspect/blob/main/notebooks/01_inspect_truthfulqa.ipynb) **Main experiment**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asadullah-moghul/qwen-truthfulqa-inspect/blob/main/notebooks/02_under_the_hood_arc.ipynb) Under the hood (optional)

1. Click a badge above. The notebook opens in Colab directly from this repo (use **File → Save a copy in Drive** to keep your edits).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run cell by cell. `LIMIT = 50` by default; set `LIMIT = None` for all 817 questions.
4. To try a bigger overseer, set `MODEL = "hf/Qwen/Qwen2.5-1.5B-Instruct"`.

No API keys and no Hugging Face token are needed: Qwen is ungated, and everything runs on the free GPU.

**Known gotchas**
- The install cell prints a pip "dependency resolver" error about `bigframes` / `gcsfs`. It's harmless; continue.
- Inspect's Hugging Face provider passes `temperature=0` straight to `transformers`, which rejects it. The notebook uses `model_args={"do_sample": False}` (greedy decoding) instead, so runs are reproducible.
- Inside notebooks Inspect needs `ipywidgets`; the install cell includes it.

## What's in here

| Path | What it is |
|---|---|
| `notebooks/01_inspect_truthfulqa.ipynb` | **The main experiment.** Inspect + `inspect_evals` on TruthfulQA and ARC, plus diagnostics: accuracy with bootstrap CI, the correct chance baseline, format compliance, position bias, and the misconceptions it picked. |
| `notebooks/02_under_the_hood_arc.ipynb` | Optional. The same kind of eval written by hand (~40 lines, no framework), scoring by reading letter probabilities instead of parsing text. Adds **calibration**: does a 90%-confident answer come out right 90% of the time? |
| `results/*.eval` | The raw Inspect logs: every prompt, reply and score. Open them with `inspect view --log-dir results` or the **Inspect AI** VS Code extension. |
| `results/summary_inspect.json` | The numbers above, machine-readable. |

## Why Inspect (vs. the other repos from the Week 3 slides)

- **Inspect** runs local Hugging Face models (`hf/` prefix) and ships 100+ benchmarks in `inspect_evals`, including AgentDojo. **[ControlArena](https://github.com/UKGovernmentBEIS/control-arena)** (UK AISI + Redwood Research), the AI Control library, is built on it.
- **AgentDojo** tests multi-step tool use; a 0.5B model can't do that reliably. It's worth revisiting with stronger models (it's already inside `inspect_evals`).
- **HarmBench** grades with a 13B classifier that won't fit a free T4.
- **simple-evals** targets API models and needs an OpenAI key for several graders.

## Next steps

- Full 817-question TruthfulQA run.
- 0.5B vs 1.5B: does a bigger overseer get fooled less? (A first look at the capability-gap question.)
- Calibration via notebook 02: does the model know when it's guessing?
- Position-bias control: rerun with the options rotated.
