# MAI 600 Module 5 — Local Model Bake-Off

**Student:** Giovanna Zeglio  
**Project option:** Option 2 — Multi-Model Comparison  
**Use case:** Local assistant for interior design and commercial furniture sales

## Project overview

This project compares two small open-weight language models running locally through Ollama. The goal is to determine which model provides the better practical balance of response quality, speed, memory use, privacy, and hardware fit for routine interior design and commercial furniture work.

The fixed prompt set tests seven behaviors: summarization, classification, structured JSON, domain terminology, resistance to unsupported claims, arithmetic reasoning, and professional tone. All scenarios are fictional and non-sensitive. Both models received the same prompts and generation settings.

## Problem and use case

An interior design sales consultant could use a local assistant to summarize client notes, classify inquiries, organize project requirements, draft customer communication, and complete preliminary furniture calculations. Expected outputs are short, clear, accurate, professionally formatted, and grounded only in the information provided.

Local inference may be useful because project notes can contain room dimensions, budgets, product requirements, and internal planning information. Running the model on the organization’s computer supports privacy, offline access, and predictable operating cost. Important risks include inaccurate calculations, invented product specifications, weak instruction following, and unsupported promises. Human review is required before client communication, purchasing, pricing, warranty, certification, or schedule decisions.

## Runtime and hardware

| Item | Tested configuration |
|---|---|
| Runtime | Ollama 0.32.6 |
| Operating system | Windows 11 |
| CPU | Intel Core Ultra 9 275HX |
| RAM | 31.42 GB (approximately 32 GB) |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| VRAM | 16,303 MiB |
| Driver / CUDA | NVIDIA driver 577.03 / CUDA 12.9 |
| Model storage | Ollama default model directory on Windows |
| Installation notes | Ollama and both models installed successfully. One pasted command initially failed because multiple commands were combined, then succeeded when entered correctly. |

## Models tested

| Model | Family | Parameters | Quantization | Download size | Runtime |
|---|---|---:|---|---:|---|
| `llama3.2:3b` | Llama | 3B | Ollama default package; exact level not separately verified | 2.0 GB | Ollama |
| `qwen3:4b-instruct` | Qwen | 4B | Q4_K_M | 2.5 GB | Ollama |

Model pages: [Llama 3.2 on Ollama](https://ollama.com/library/llama3.2) and [Qwen3 on Ollama](https://ollama.com/library/qwen3).

## Prompt set summary

| ID | Behavior tested | Expected behavior |
|---|---|---|
| P1 | Summarization | Include all supplied project facts in no more than 80 words. |
| P2 | Classification | Assign exactly one allowed label to each inquiry. |
| P3 | Structured output | Return valid JSON with the exact requested schema. |
| P4 | Domain-specific language | Use office-furniture terminology correctly and include a measurement caveat. |
| P5 | Hallucination sensitivity | Refuse to invent missing product specifications and request a reliable source. |
| P6 | Reasoning | Calculate the order total and remaining budget correctly. |
| P7 | Tone/persona | Produce an empathetic professional reply without unsupported promises. |

The complete wording is in [`prompt_set.md`](prompt_set.md).

## Benchmark method

Each prompt was sent to Ollama’s local `/api/generate` endpoint once per model with temperature `0.2`, seed `42`, and a maximum of `350` generated tokens. The notebook recorded wall-clock response time and used Ollama’s evaluation count and duration to calculate tokens per second. It sampled `nvidia-smi` every 0.1 seconds during each request to estimate peak total GPU memory use.

The GPU figure is total memory observed during the request, not an isolated model-only allocation. Each model was unloaded after its response (`keep_alive: 0s`), so the recorded response time includes loading overhead and represents a cold-start style test. These design choices make the comparison reproducible but do not establish production throughput or statistical significance.

## Results summary

| Model | Avg. response time | Avg. tokens/sec | Max peak GPU memory | Overall quality |
|---|---:|---:|---:|---:|
| Llama 3.2 3B | 5.57 s | 152.86 | 3,375 MiB | 4.04 / 5 |
| Qwen3 4B Instruct | 5.77 s | 118.34 | 3,971 MiB | **4.76 / 5** |

Llama generated approximately 29% more tokens per second and used 596 MiB less peak GPU memory. Qwen was only 0.20 seconds slower on average and used well below the laptop’s 16,303 MiB VRAM capacity. The response-time difference was not important in normal interactive use.

Quality changed the decision. Qwen used the domain terms correctly, refused to confirm unverified product specifications, and calculated the furniture total accurately. Llama performed well on straightforward summary, classification, and JSON tasks, but it invented a certification in P5 and made multiple arithmetic errors in P6. Qwen was not perfect: its P7 response invented a logistical explanation and promised an unconfirmed update deadline. Both models therefore require human review.

Detailed measurements are in [`benchmark_results.csv`](benchmark_results.csv) and the full rubric is in [`quality_scores.csv`](quality_scores.csv).

## Speed, quality, memory, model-size, and quantization trade-off

The accepted trade-off is modestly lower speed and higher memory use in exchange for stronger accuracy, grounding, domain understanding, and instruction following. Qwen’s 2.5 GB Q4_K_M package was 0.5 GB larger than the tested Llama package and used about 0.6 GB more peak GPU memory, but the laptop had ample capacity. This was a model-family comparison, not a controlled quantization sweep, so the experiment cannot isolate how much of the result was caused by quantization rather than model architecture, size, or training.

## Final recommendation

**Select Qwen3 4B Instruct for this local use case.** It is practical on the tested Windows laptop and provides the better balance for summaries, classifications, structured project notes, domain explanations, and first drafts. The small performance cost is justified by its 0.72-point quality advantage.

Llama 3.2 3B remains useful where maximum generation speed and lower memory use matter and the task is simple and low risk. A hosted API may still be preferable for complex reasoning, broader knowledge, enterprise support, or high-consequence client, financial, specification, and schedule decisions. Local deployment improves control and privacy; it does not remove the need for reliable source documents and professional judgment.

## Repository contents

The six required submission files are stored at the repository root: `README.md`, `prompt_set.md`, `benchmark_results.csv`, `quality_scores.csv`, `model_selection_memo.md`, and `ai_usage_disclosure.md`. The repository also includes the executed `local_model_benchmark.ipynb`, quality chart, sample-output review, and supporting benchmark files.

## Reproduction

1. Install Ollama on a Windows computer with sufficient free storage and memory.
2. Download the models:

   ```powershell
   ollama pull llama3.2:3b
   ollama pull qwen3:4b-instruct
   ```

3. Open Ollama and run [`local_model_benchmark.ipynb`](local_model_benchmark.ipynb) in Jupyter.
4. Review every output before assigning quality scores. Scores in this repository represent the submitted human evaluation of the recorded run.

## AI usage disclosure

ChatGPT and OpenAI Codex were used as learning and support tools to organize the project, draft safe fictional prompts, help create benchmark code, and assemble the deliverables. No hosted model API was used to generate the benchmark results: both tested models ran locally through Ollama. The student installed the runtime, downloaded and ran the models, checked the computer specifications, reviewed the responses, verified the results, assigned the quality scores, and selected the final model. See [`ai_usage_disclosure.md`](ai_usage_disclosure.md) for the complete disclosure.
