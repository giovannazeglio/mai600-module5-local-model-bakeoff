# AI Tool Usage Disclosure

## AI tools used

- [x] ChatGPT
- [ ] Claude
- [ ] Gemini
- [ ] Claude Code
- [ ] GitHub Copilot
- [x] OpenAI Codex
- [ ] Other: None

## Local tools used

- [x] Ollama
- [ ] LM Studio
- [ ] llama.cpp
- [ ] AU lab environment
- [x] Other: Jupyter Notebook, Python, `requests`, `pandas`, `matplotlib`, and `nvidia-smi`

## API usage

- [x] No hosted API used for the benchmark
- [ ] Hosted API used for comparison: Not applicable

The notebook called Ollama’s local API at `http://localhost:11434/api/generate`. This endpoint ran on the same Windows computer and was not a hosted model API.

## How I used AI

- I used ChatGPT and OpenAI Codex as learning and support tools to interpret the assignment, organize the notebook and repository, draft fictional non-sensitive test prompts, and create Python benchmark code.
- AI assisted with summarizing the measured results, identifying trade-offs, and drafting repository documentation.
- The local models generated the responses evaluated in the experiment.

## Prompts used

- Summarize fictional Northstar Creative Studio project notes in 80 words or fewer.
- Classify five fictional customer inquiries using five allowed labels.
- Convert a fictional office-refresh description into an exact JSON schema.
- Explain five commercial office-furniture concepts in one professional paragraph.
- Respond safely when fictional chair warranty, certification, and capacity claims cannot be verified.
- Calculate a fictional furniture subtotal, tax, installation cost, final total, and remaining budget.
- Rewrite a fictional delivery complaint in an empathetic professional tone without unsupported promises.

The complete prompt wording is preserved in [`prompt_set.md`](prompt_set.md).

## What I verified myself

- I installed Ollama and confirmed the installed version.
- I checked my Windows operating system, CPU, RAM, GPU, VRAM, NVIDIA driver, and CUDA information.
- I downloaded both local models and confirmed their local file sizes.
- I ran all 14 tests on my own computer.
- I reviewed the outputs against the prompt facts and instructions.
- I verified the benchmark numbers and the expected arithmetic result for P6.
- I assigned the 1–5 quality scores and selected the final model based on the measured evidence.

## Failures and limitations

- One Ollama command initially failed because multiple commands were pasted together; the model downloaded correctly after the command was re-entered by itself.
- Each prompt/model combination was run once, so the experiment does not measure run-to-run variation or statistical significance.
- `keep_alive: 0s` unloaded the model after each request, so the response-time values include model-loading overhead.
- Peak GPU memory is the total memory observed through `nvidia-smi`, not a perfectly isolated model allocation.
- The 350-token limit truncated both P6 responses; Qwen had already produced the correct subtotal, tax, and final total, but did not reach the remaining-budget line.
- The exact quantization of the default Llama Ollama package was not separately verified.
- This comparison does not prove that Qwen will be better for every prompt, hardware configuration, runtime version, or professional use case.
- Both models produced at least one output that required correction, so human review remains necessary.

I confirm that AI was used as a learning and support tool, not as a replacement for my own work.

