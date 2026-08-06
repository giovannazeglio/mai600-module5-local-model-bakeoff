{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "fe919a8d",
      "metadata": {},
      "source": [
        "# Module 5 Assignment: Local Open-Weight LLM Benchmark\n",
        "\n",
        "**Student:** Giovanna Zeglio  \n",
        "**Project option:** Option 2 — Multi-Model Comparison  \n",
        "**Models:** Llama 3.2 3B and Qwen3 4B Instruct  \n",
        "**Runtime:** Ollama on Windows\n",
        "\n",
        "## Introduction\n",
        "\n",
        "This project compares two small open-weight language models running locally through Ollama. The benchmark is based on a fixed set of fictional prompts related to interior design and commercial furniture sales. These tasks reflect the kind of work an assistant might support in a real business setting, including summarizing client notes, classifying requests, organizing project information, using industry terminology, avoiding unsupported claims, completing a cost calculation, and adjusting the tone of a customer response. The purpose is not simply to identify which model produces the strongest answer. It is to determine whether either model offers a practical balance of speed, quality, privacy, hardware use, and ease of operation for this use case.",
        "\n",
        "**Status:** Completed with locally measured benchmark results and human quality evaluation.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "3a777cb8",
      "metadata": {},
      "source": [
        "## Part 1: Define the Use Case\n",
        "\n",
        "The selected use case is a local writing and decision-support assistant for fictional interior design and commercial furniture projects. An interior design sales consultant could use the system to summarize client notes, organize requirements, classify incoming requests, draft professional communication, and perform simple project calculations. The expected outputs are short, clear, accurate, and professionally formatted responses that remain grounded in the information supplied in each prompt.\n",
        "\n",
        "A local model may be useful because project notes can include room dimensions, budgets, product requirements, and internal planning information. Keeping inference on the organization’s own computer may provide greater privacy, offline access, and predictable operating costs. It also reduces dependence on an external API for routine tasks. The main risks are inaccurate calculations, failure to follow a requested format, weak understanding of industry language, unsupported product claims, and slower performance on limited hardware. Human review would still be required before any output is sent to a client or used for a purchasing decision."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "143b8763",
      "metadata": {},
      "source": [
        "## Part 2: Local Runtime and Hardware\n",
        "\n",
        "| Item | Response |\n",
        "|---|---|\n",
        "| Runtime used | Ollama 0.32.6 |\n",
        "| Operating system | Windows |\n",
        "| CPU | Intel Core Ultra 9 275HX |\n",
        "| RAM | 31.42 GB (approximately 32 GB) |\n",
        "| GPU / VRAM | NVIDIA GeForce RTX 5080 Laptop GPU / 16,303 MiB |\n",
        "| NVIDIA driver / CUDA | Driver 577.03 / CUDA 12.9 |\n",
        "| Model storage location | Ollama default model directory on the local Windows system |\n",
        "| Installation issues | Ollama installed successfully. One command initially failed because it was pasted more than once, but it worked after the command was entered correctly. |\n",
        "\n",
        "Ollama was selected because it provides a simple local runtime, a command-line interface, and a local API that exposes timing and token-generation information."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "e1b1de63",
      "metadata": {},
      "source": [
        "## Part 3: Models Selected\n",
        "\n",
        "| Model name | Family | Parameters | Quantization | Runtime | Local file size | Notes |\n",
        "|---|---|---:|---|---|---:|---|\n",
        "| `llama3.2:3b` | Llama | 3B | Ollama default package | Ollama | 2.0 GB | Small multilingual instruction model |\n",
        "| `qwen3:4b-instruct` | Qwen | 4B | Q4_K_M package | Ollama | 2.5 GB | Small instruction-focused model |\n",
        "\n",
        "Model pages: [Llama 3.2](https://ollama.com/library/llama3.2) and [Qwen3](https://ollama.com/library/qwen3). Both models are small enough to run comfortably on the available laptop hardware, while their different model families make the comparison meaningful."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6a999b8a",
      "metadata": {},
      "source": [
        "## Environment Setup\n",
        "\n",
        "Run this notebook on the same Windows computer where Ollama and both models are installed. Ollama must be open before the benchmark begins."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "id": "576f0124",
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Note: you may need to restart the kernel to use updated packages.\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 25.2 -> 26.2.1\n",
            "[notice] To update, run: C:\\Users\\giova\\AppData\\Local\\Programs\\Python\\Python314\\python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "%pip install requests pandas matplotlib -q"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "id": "f0fa1c43",
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Python: 3.14.0\n",
            "Platform: Windows-11-10.0.26200-SP0\n",
            "Models selected: ['llama3.2:3b', 'qwen3:4b-instruct']\n"
          ]
        }
      ],
      "source": [
        "import json\n",
        "import platform\n",
        "import subprocess\n",
        "import threading\n",
        "import time\n",
        "from pathlib import Path\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import requests\n",
        "from IPython.display import Markdown, display\n",
        "\n",
        "OLLAMA_URL = \"http://localhost:11434/api/generate\"\n",
        "MODELS = [\"llama3.2:3b\", \"qwen3:4b-instruct\"]\n",
        "OUTPUT_DIR = Path(\"benchmark_outputs\")\n",
        "OUTPUT_DIR.mkdir(exist_ok=True)\n",
        "\n",
        "print(\"Python:\", platform.python_version())\n",
        "print(\"Platform:\", platform.platform())\n",
        "print(\"Models selected:\", MODELS)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "id": "d145d8e7",
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Ollama API response: {'version': '0.32.6'}\n",
            "Installed models: ['qwen3:4b-instruct', 'llama3.2:3b']\n"
          ]
        }
      ],
      "source": [
        "# Confirm that Ollama is available and that both models are installed.\n",
        "version_response = requests.get(\"http://localhost:11434/api/version\", timeout=10)\n",
        "version_response.raise_for_status()\n",
        "print(\"Ollama API response:\", version_response.json())\n",
        "\n",
        "tags_response = requests.get(\"http://localhost:11434/api/tags\", timeout=10)\n",
        "tags_response.raise_for_status()\n",
        "installed_models = [item[\"name\"] for item in tags_response.json().get(\"models\", [])]\n",
        "print(\"Installed models:\", installed_models)\n",
        "\n",
        "for model in MODELS:\n",
        "    assert model in installed_models, f\"Missing model: {model}. Run 'ollama pull {model}' in PowerShell.\""
      ]
    },
    {
      "cell_type": "markdown",
      "id": "85a0cd37",
      "metadata": {},
      "source": [
        "## Part 4: Fixed Prompt Set\n",
        "\n",
        "All information below is fictional and non-sensitive. The same prompt text, temperature, random seed, and output limit will be used for both models."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "153b61d0",
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>prompt_id</th>\n",
              "      <th>prompt_type</th>\n",
              "      <th>prompt_text</th>\n",
              "      <th>expected_behavior</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>P1</td>\n",
              "      <td>Summarization</td>\n",
              "      <td>Summarize the following fictional client notes in no more than 80 words. Include the project size, requested furniture, priorities, budg...</td>\n",
              "      <td>Summarize the supplied facts accurately in no more than 80 words.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>P2</td>\n",
              "      <td>Classification</td>\n",
              "      <td>Classify each fictional customer inquiry using exactly one of these labels: SPACE_PLANNING, PRODUCT_QUESTION, DELIVERY_ISSUE, PRICE_REQU...</td>\n",
              "      <td>Assign exactly one allowed label to each inquiry.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>P3</td>\n",
              "      <td>Structured output</td>\n",
              "      <td>Convert the following fictional project information into valid JSON. Use exactly these keys: project_name, room_width_ft, room_length_ft...</td>\n",
              "      <td>Return valid JSON with the exact requested keys and no extra commentary.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>P4</td>\n",
              "      <td>Domain-specific</td>\n",
              "      <td>You are supporting a fictional commercial office furniture project. Explain how benching workstations, mobile pedestals, acoustic desk s...</td>\n",
              "      <td>Use relevant interior design and furniture terminology while identifying assumptions.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>P5</td>\n",
              "      <td>Hallucination-sensitive</td>\n",
              "      <td>A client asks whether the fictional Aurelia X200 task chair has a 12-year warranty, GREENGUARD Gold certification, and a 300-pound weigh...</td>\n",
              "      <td>Refuse to invent missing specifications and request a reliable source.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>P6</td>\n",
              "      <td>Reasoning</td>\n",
              "      <td>Calculate the total for this fictional furniture order: 18 desks at $680 each, 18 task chairs at $295 each, and 18 mobile pedestals at $...</td>\n",
              "      <td>Calculate a final total of $23,773.10 and remaining budget of $1,226.90.</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>P7</td>\n",
              "      <td>Tone/persona</td>\n",
              "      <td>Rewrite the following fictional customer complaint as a professional reply from an interior design sales consultant. Use an empathetic b...</td>\n",
              "      <td>Create an empathetic and professional reply without unsupported promises.</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  prompt_id              prompt_type  \\\n",
              "0        P1            Summarization   \n",
              "1        P2           Classification   \n",
              "2        P3        Structured output   \n",
              "3        P4          Domain-specific   \n",
              "4        P5  Hallucination-sensitive   \n",
              "5        P6                Reasoning   \n",
              "6        P7             Tone/persona   \n",
              "\n",
              "                                                                                                                                   prompt_text  \\\n",
              "0  Summarize the following fictional client notes in no more than 80 words. Include the project size, requested furniture, priorities, budg...   \n",
              "1  Classify each fictional customer inquiry using exactly one of these labels: SPACE_PLANNING, PRODUCT_QUESTION, DELIVERY_ISSUE, PRICE_REQU...   \n",
              "2  Convert the following fictional project information into valid JSON. Use exactly these keys: project_name, room_width_ft, room_length_ft...   \n",
              "3  You are supporting a fictional commercial office furniture project. Explain how benching workstations, mobile pedestals, acoustic desk s...   \n",
              "4  A client asks whether the fictional Aurelia X200 task chair has a 12-year warranty, GREENGUARD Gold certification, and a 300-pound weigh...   \n",
              "5  Calculate the total for this fictional furniture order: 18 desks at $680 each, 18 task chairs at $295 each, and 18 mobile pedestals at $...   \n",
              "6  Rewrite the following fictional customer complaint as a professional reply from an interior design sales consultant. Use an empathetic b...   \n",
              "\n",
              "                                                                       expected_behavior  \n",
              "0                      Summarize the supplied facts accurately in no more than 80 words.  \n",
              "1                                      Assign exactly one allowed label to each inquiry.  \n",
              "2               Return valid JSON with the exact requested keys and no extra commentary.  \n",
              "3  Use relevant interior design and furniture terminology while identifying assumptions.  \n",
              "4                 Refuse to invent missing specifications and request a reliable source.  \n",
              "5               Calculate a final total of $23,773.10 and remaining budget of $1,226.90.  \n",
              "6              Create an empathetic and professional reply without unsupported promises.  "
            ]
          },
          "execution_count": 17,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "PROMPTS = [\n",
        "    {\n",
        "        \"prompt_id\": \"P1\",\n",
        "        \"prompt_type\": \"Summarization\",\n",
        "        \"expected_behavior\": \"Summarize the supplied facts accurately in no more than 80 words.\",\n",
        "        \"prompt_text\": \"\"\"Summarize the following fictional client notes in no more than 80 words. Include the project size, requested furniture, priorities, budget, and deadline. Do not add information.\\n\\nClient: Northstar Creative Studio. Office size: 2,400 square feet. Requested furniture: 18 height-adjustable desks, 18 ergonomic task chairs, two six-person meeting tables, and four mobile storage pedestals. Priorities: acoustic comfort, flexible collaboration, and a clear central circulation path. Furniture budget: $28,000 before tax. Requested installation date: October 16, 2026.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P2\",\n",
        "        \"prompt_type\": \"Classification\",\n",
        "        \"expected_behavior\": \"Assign exactly one allowed label to each inquiry.\",\n",
        "        \"prompt_text\": \"\"\"Classify each fictional customer inquiry using exactly one of these labels: SPACE_PLANNING, PRODUCT_QUESTION, DELIVERY_ISSUE, PRICE_REQUEST, or WARRANTY_REQUEST. Return only a two-column Markdown table with Inquiry ID and Label.\\n\\nI1: Can twelve workstations fit while keeping the main aisle open?\\nI2: Does this chair include adjustable lumbar support?\\nI3: Our conference table arrived without the hardware box.\\nI4: Please send pricing for twenty desks and chairs.\\nI5: The height-adjustment mechanism stopped working after six months.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P3\",\n",
        "        \"prompt_type\": \"Structured output\",\n",
        "        \"expected_behavior\": \"Return valid JSON with the exact requested keys and no extra commentary.\",\n",
        "        \"prompt_text\": \"\"\"Convert the following fictional project information into valid JSON. Use exactly these keys: project_name, room_width_ft, room_length_ft, workstation_count, finish, budget_usd, and priority. Return JSON only, without Markdown.\\n\\nProject name: Harbor Point Office Refresh; room width: 24 feet; room length: 36 feet; workstations: 12; finish: white laminate with silver frames; budget: $21,500; priority: complete installation before staff return.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P4\",\n",
        "        \"prompt_type\": \"Domain-specific\",\n",
        "        \"expected_behavior\": \"Use relevant interior design and furniture terminology while identifying assumptions.\",\n",
        "        \"prompt_text\": \"\"\"You are supporting a fictional commercial office furniture project. Explain how benching workstations, mobile pedestals, acoustic desk screens, task chairs, and a clear circulation path could work together in an open office. Write one professional paragraph of 120 to 160 words. Use the terms correctly, focus on function, and state that final dimensions must be verified in a measured floor plan. Do not claim code compliance.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P5\",\n",
        "        \"prompt_type\": \"Hallucination-sensitive\",\n",
        "        \"expected_behavior\": \"Refuse to invent missing specifications and request a reliable source.\",\n",
        "        \"prompt_text\": \"\"\"A client asks whether the fictional Aurelia X200 task chair has a 12-year warranty, GREENGUARD Gold certification, and a 300-pound weight capacity. No product sheet, manufacturer name, or verified specifications are available. Answer in 70 to 100 words. Clearly separate what is known from what cannot be verified, do not invent product facts, and recommend the next verification step.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P6\",\n",
        "        \"prompt_type\": \"Reasoning\",\n",
        "        \"expected_behavior\": \"Calculate a final total of $23,773.10 and remaining budget of $1,226.90.\",\n",
        "        \"prompt_text\": \"\"\"Calculate the total for this fictional furniture order: 18 desks at $680 each, 18 task chairs at $295 each, and 18 mobile pedestals at $210 each. Sales tax is 7% and applies only to the furniture subtotal. Installation is $950 and is not taxable. The total budget is $25,000. Show the furniture subtotal, tax, installation, final total, and amount remaining or over budget. Round currency to two decimals and briefly explain each step.\"\"\"\n",
        "    },\n",
        "    {\n",
        "        \"prompt_id\": \"P7\",\n",
        "        \"prompt_type\": \"Tone/persona\",\n",
        "        \"expected_behavior\": \"Create an empathetic and professional reply without unsupported promises.\",\n",
        "        \"prompt_text\": \"\"\"Rewrite the following fictional customer complaint as a professional reply from an interior design sales consultant. Use an empathetic but calm tone, keep the response between 100 and 130 words, acknowledge the concern, explain that the delivery record will be reviewed, and provide a clear next step. Do not promise a refund or a delivery date that has not been confirmed.\\n\\nComplaint: Half of the chairs arrived today, two cartons look damaged, and nobody told us the order would be split. We need an answer now because our staff returns next week.\"\"\"\n",
        "    },\n",
        "]\n",
        "\n",
        "prompt_table = pd.DataFrame(PROMPTS)[[\"prompt_id\", \"prompt_type\", \"prompt_text\", \"expected_behavior\"]]\n",
        "pd.set_option(\"display.max_colwidth\", 140)\n",
        "prompt_table"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "42638960",
      "metadata": {},
      "source": [
        "## Part 5: Benchmark Performance\n",
        "\n",
        "The function below sends each prompt to the local Ollama API and records both wall-clock time and the timing values returned by Ollama. Tokens per second is calculated as generated tokens divided by evaluation duration. GPU memory is sampled during each request with `nvidia-smi`; this measurement represents total GPU memory in use at the observed peak, not a perfectly isolated model allocation. Temperature is fixed at 0.2, seed at 42, and the output limit at 350 tokens to make the comparison more consistent."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "515eab05",
      "metadata": {},
      "outputs": [],
      "source": [
        "def get_gpu_memory_mib():\n",
        "    try:\n",
        "        completed = subprocess.run(\n",
        "            [\"nvidia-smi\", \"--query-gpu=memory.used\", \"--format=csv,noheader,nounits\"],\n",
        "            capture_output=True, text=True, check=True\n",
        "        )\n",
        "        values = [float(value.strip()) for value in completed.stdout.splitlines() if value.strip()]\n",
        "        return max(values) if values else None\n",
        "    except Exception:\n",
        "        return None\n",
        "\n",
        "\n",
        "def benchmark_prompt(model, prompt_record):\n",
        "    memory_samples = []\n",
        "    stop_event = threading.Event()\n",
        "\n",
        "    def sample_memory():\n",
        "        while not stop_event.is_set():\n",
        "            value = get_gpu_memory_mib()\n",
        "            if value is not None:\n",
        "                memory_samples.append(value)\n",
        "            time.sleep(0.1)\n",
        "\n",
        "    baseline_memory = get_gpu_memory_mib()\n",
        "    sampler = threading.Thread(target=sample_memory, daemon=True)\n",
        "    sampler.start()\n",
        "\n",
        "    payload = {\n",
        "        \"model\": model,\n",
        "        \"prompt\": prompt_record[\"prompt_text\"],\n",
        "        \"stream\": False,\n",
        "        \"keep_alive\": \"0s\",\n",
        "        \"options\": {\"temperature\": 0.2, \"seed\": 42, \"num_predict\": 350},\n",
        "    }\n",
        "\n",
        "    start = time.perf_counter()\n",
        "    try:\n",
        "        api_response = requests.post(OLLAMA_URL, json=payload, timeout=600)\n",
        "        api_response.raise_for_status()\n",
        "        result = api_response.json()\n",
        "    finally:\n",
        "        wall_time = time.perf_counter() - start\n",
        "        stop_event.set()\n",
        "        sampler.join(timeout=2)\n",
        "\n",
        "    eval_count = result.get(\"eval_count\")\n",
        "    eval_duration_ns = result.get(\"eval_duration\")\n",
        "    tokens_per_second = None\n",
        "    if eval_count is not None and eval_duration_ns:\n",
        "        tokens_per_second = eval_count / (eval_duration_ns / 1_000_000_000)\n",
        "\n",
        "    peak_memory = max(memory_samples) if memory_samples else None\n",
        "    memory_delta = None\n",
        "    if peak_memory is not None and baseline_memory is not None:\n",
        "        memory_delta = max(0, peak_memory - baseline_memory)\n",
        "\n",
        "    response_text = result.get(\"response\", \"\").strip()\n",
        "    return {\n",
        "        \"Model\": model,\n",
        "        \"Prompt ID\": prompt_record[\"prompt_id\"],\n",
        "        \"Prompt Type\": prompt_record[\"prompt_type\"],\n",
        "        \"Response Time (s)\": round(wall_time, 3),\n",
        "        \"Ollama Total Duration (s)\": round(result.get(\"total_duration\", 0) / 1_000_000_000, 3),\n",
        "        \"Load Duration (s)\": round(result.get(\"load_duration\", 0) / 1_000_000_000, 3),\n",
        "        \"Tokens/sec\": round(tokens_per_second, 2) if tokens_per_second is not None else \"Not available in my runtime\",\n",
        "        \"Peak GPU Memory (MiB)\": round(peak_memory, 1) if peak_memory is not None else \"Not available in my runtime\",\n",
        "        \"GPU Memory Increase (MiB)\": round(memory_delta, 1) if memory_delta is not None else \"Not available in my runtime\",\n",
        "        \"Output Length (tokens)\": eval_count if eval_count is not None else \"Not available in my runtime\",\n",
        "        \"Output Length (words)\": len(response_text.split()),\n",
        "        \"Response\": response_text,\n",
        "    }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "eecadb4b",
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Starting benchmark for llama3.2:3b\n",
            "  Running P1... done (5.197 seconds, 165.05 tokens/sec)\n",
            "  Running P2... done (4.921 seconds, 164.81 tokens/sec)\n",
            "  Running P3... done (5.053 seconds, 165.0 tokens/sec)\n",
            "  Running P4... done (5.729 seconds, 140.14 tokens/sec)\n",
            "  Running P5... done (5.348 seconds, 154.17 tokens/sec)\n",
            "  Running P6... done (7.093 seconds, 140.78 tokens/sec)\n",
            "  Running P7... done (5.637 seconds, 140.06 tokens/sec)\n",
            "\n",
            "Starting benchmark for qwen3:4b-instruct\n",
            "  Running P1... done (5.171 seconds, 127.83 tokens/sec)\n",
            "  Running P2... done (5.008 seconds, 124.05 tokens/sec)\n",
            "  Running P3... done (5.225 seconds, 125.26 tokens/sec)\n",
            "  Running P4... done (5.92 seconds, 112.39 tokens/sec)\n",
            "  Running P5... done (5.419 seconds, 116.29 tokens/sec)\n",
            "  Running P6... done (7.773 seconds, 109.7 tokens/sec)\n",
            "  Running P7... done (5.849 seconds, 112.85 tokens/sec)\n",
            "\n",
            "Benchmark completed and results saved.\n"
          ]
        }
      ],
      "source": [
        "# Run all 14 tests: 7 prompts x 2 models. This may take several minutes.\n",
        "benchmark_records = []\n",
        "\n",
        "for model in MODELS:\n",
        "    print(f\"\\nStarting benchmark for {model}\")\n",
        "    for prompt_record in PROMPTS:\n",
        "        print(f\"  Running {prompt_record['prompt_id']}...\", end=\" \" )\n",
        "        record = benchmark_prompt(model, prompt_record)\n",
        "        benchmark_records.append(record)\n",
        "        print(f\"done ({record['Response Time (s)']} seconds, {record['Tokens/sec']} tokens/sec)\")\n",
        "\n",
        "results_df = pd.DataFrame(benchmark_records)\n",
        "results_df.to_csv(OUTPUT_DIR / \"benchmark_results.csv\", index=False)\n",
        "with open(OUTPUT_DIR / \"benchmark_results.json\", \"w\", encoding=\"utf-8\") as file:\n",
        "    json.dump(benchmark_records, file, indent=2, ensure_ascii=False)\n",
        "\n",
        "print(\"\\nBenchmark completed and results saved.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "id": "919cbbc2",
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Model</th>\n",
              "      <th>Prompt ID</th>\n",
              "      <th>Response Time (s)</th>\n",
              "      <th>Tokens/sec</th>\n",
              "      <th>Peak GPU Memory (MiB)</th>\n",
              "      <th>GPU Memory Increase (MiB)</th>\n",
              "      <th>Output Length (tokens)</th>\n",
              "      <th>Output Length (words)</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P1</td>\n",
              "      <td>5.197</td>\n",
              "      <td>165.05</td>\n",
              "      <td>3371.0</td>\n",
              "      <td>2661.0</td>\n",
              "      <td>92</td>\n",
              "      <td>52</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P2</td>\n",
              "      <td>4.921</td>\n",
              "      <td>164.81</td>\n",
              "      <td>3371.0</td>\n",
              "      <td>2661.0</td>\n",
              "      <td>51</td>\n",
              "      <td>36</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P3</td>\n",
              "      <td>5.053</td>\n",
              "      <td>165.00</td>\n",
              "      <td>3375.0</td>\n",
              "      <td>2665.0</td>\n",
              "      <td>65</td>\n",
              "      <td>25</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P4</td>\n",
              "      <td>5.729</td>\n",
              "      <td>140.14</td>\n",
              "      <td>3373.0</td>\n",
              "      <td>2663.0</td>\n",
              "      <td>160</td>\n",
              "      <td>137</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P5</td>\n",
              "      <td>5.348</td>\n",
              "      <td>154.17</td>\n",
              "      <td>3373.0</td>\n",
              "      <td>2663.0</td>\n",
              "      <td>110</td>\n",
              "      <td>89</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P6</td>\n",
              "      <td>7.093</td>\n",
              "      <td>140.78</td>\n",
              "      <td>3375.0</td>\n",
              "      <td>2665.0</td>\n",
              "      <td>350</td>\n",
              "      <td>189</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>llama3.2:3b</td>\n",
              "      <td>P7</td>\n",
              "      <td>5.637</td>\n",
              "      <td>140.06</td>\n",
              "      <td>3371.0</td>\n",
              "      <td>2661.0</td>\n",
              "      <td>141</td>\n",
              "      <td>124</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P1</td>\n",
              "      <td>5.171</td>\n",
              "      <td>127.83</td>\n",
              "      <td>3967.0</td>\n",
              "      <td>3257.0</td>\n",
              "      <td>78</td>\n",
              "      <td>36</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P2</td>\n",
              "      <td>5.008</td>\n",
              "      <td>124.05</td>\n",
              "      <td>3971.0</td>\n",
              "      <td>3261.0</td>\n",
              "      <td>57</td>\n",
              "      <td>32</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P3</td>\n",
              "      <td>5.225</td>\n",
              "      <td>125.26</td>\n",
              "      <td>3969.0</td>\n",
              "      <td>3259.0</td>\n",
              "      <td>80</td>\n",
              "      <td>27</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P4</td>\n",
              "      <td>5.920</td>\n",
              "      <td>112.39</td>\n",
              "      <td>3967.0</td>\n",
              "      <td>3257.0</td>\n",
              "      <td>150</td>\n",
              "      <td>122</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P5</td>\n",
              "      <td>5.419</td>\n",
              "      <td>116.29</td>\n",
              "      <td>3967.0</td>\n",
              "      <td>3257.0</td>\n",
              "      <td>100</td>\n",
              "      <td>74</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P6</td>\n",
              "      <td>7.773</td>\n",
              "      <td>109.70</td>\n",
              "      <td>3971.0</td>\n",
              "      <td>3261.0</td>\n",
              "      <td>350</td>\n",
              "      <td>162</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>qwen3:4b-instruct</td>\n",
              "      <td>P7</td>\n",
              "      <td>5.849</td>\n",
              "      <td>112.85</td>\n",
              "      <td>3971.0</td>\n",
              "      <td>3261.0</td>\n",
              "      <td>145</td>\n",
              "      <td>121</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                Model Prompt ID  Response Time (s)  Tokens/sec  \\\n",
              "0         llama3.2:3b        P1              5.197      165.05   \n",
              "1         llama3.2:3b        P2              4.921      164.81   \n",
              "2         llama3.2:3b        P3              5.053      165.00   \n",
              "3         llama3.2:3b        P4              5.729      140.14   \n",
              "4         llama3.2:3b        P5              5.348      154.17   \n",
              "5         llama3.2:3b        P6              7.093      140.78   \n",
              "6         llama3.2:3b        P7              5.637      140.06   \n",
              "7   qwen3:4b-instruct        P1              5.171      127.83   \n",
              "8   qwen3:4b-instruct        P2              5.008      124.05   \n",
              "9   qwen3:4b-instruct        P3              5.225      125.26   \n",
              "10  qwen3:4b-instruct        P4              5.920      112.39   \n",
              "11  qwen3:4b-instruct        P5              5.419      116.29   \n",
              "12  qwen3:4b-instruct        P6              7.773      109.70   \n",
              "13  qwen3:4b-instruct        P7              5.849      112.85   \n",
              "\n",
              "    Peak GPU Memory (MiB)  GPU Memory Increase (MiB)  Output Length (tokens)  \\\n",
              "0                  3371.0                     2661.0                      92   \n",
              "1                  3371.0                     2661.0                      51   \n",
              "2                  3375.0                     2665.0                      65   \n",
              "3                  3373.0                     2663.0                     160   \n",
              "4                  3373.0                     2663.0                     110   \n",
              "5                  3375.0                     2665.0                     350   \n",
              "6                  3371.0                     2661.0                     141   \n",
              "7                  3967.0                     3257.0                      78   \n",
              "8                  3971.0                     3261.0                      57   \n",
              "9                  3969.0                     3259.0                      80   \n",
              "10                 3967.0                     3257.0                     150   \n",
              "11                 3967.0                     3257.0                     100   \n",
              "12                 3971.0                     3261.0                     350   \n",
              "13                 3971.0                     3261.0                     145   \n",
              "\n",
              "    Output Length (words)  \n",
              "0                      52  \n",
              "1                      36  \n",
              "2                      25  \n",
              "3                     137  \n",
              "4                      89  \n",
              "5                     189  \n",
              "6                     124  \n",
              "7                      36  \n",
              "8                      32  \n",
              "9                      27  \n",
              "10                    122  \n",
              "11                     74  \n",
              "12                    162  \n",
              "13                    121  "
            ]
          },
          "execution_count": 20,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "benchmark_table = results_df[[\n",
        "    \"Model\", \"Prompt ID\", \"Response Time (s)\", \"Tokens/sec\",\n",
        "    \"Peak GPU Memory (MiB)\", \"GPU Memory Increase (MiB)\",\n",
        "    \"Output Length (tokens)\", \"Output Length (words)\"\n",
        "]].copy()\n",
        "benchmark_table"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "id": "a974b75d",
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P1 (Summarization)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Project details:\n",
              "\n",
              "* Project size: 2,400 square feet\n",
              "* Requested furniture: \n",
              "  - 18 height-adjustable desks\n",
              "  - 18 ergonomic task chairs\n",
              "  - Two six-person meeting tables\n",
              "  - Four mobile storage pedestals\n",
              "* Priorities: acoustic comfort, flexible collaboration, clear central circulation path\n",
              "* Furniture budget: $28,000 (before tax)\n",
              "* Installation date: October 16, 2026"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P2 (Classification)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "| Inquiry ID | Label |\n",
              "| --- | --- |\n",
              "| I1 | SPACE_PLANNING |\n",
              "| I2 | PRODUCT_QUESTION |\n",
              "| I3 | DELIVERY_ISSUE |\n",
              "| I4 | PRICE_REQUEST |\n",
              "| I5 | WARRANTY_REQUEST |"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P3 (Structured output)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "{\"project_name\": \"Harbor Point Office Refresh\", \"room_width_ft\": 24, \"room_length_ft\": 36, \"workstation_count\": 12, \"finish\": \"white laminate with silver frames\", \"budget_usd\": 21500, \"priority\": \"complete installation before staff return\"}"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P4 (Domain-specific)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "In an open office design, benching workstations can provide a flexible and collaborative workspace by grouping employees together around a central area. Mobile pedestals can be strategically placed to support laptops or monitors, keeping them at a comfortable height while allowing for easy movement. Acoustic desk screens can help reduce distractions and create a more focused environment, particularly in areas with high foot traffic. Task chairs can offer ergonomic comfort and support, while also providing a clear distinction between individual workspaces. A well-designed circulation path can facilitate smooth movement through the space, minimizing congestion and promoting productivity. Final dimensions must be verified in a measured floor plan to ensure a functional and efficient layout. By integrating these elements thoughtfully, an open office design can balance collaboration, focus, and comfort to create a productive and engaging work environment."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P5 (Hallucination-sensitive)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Based on the client's inquiry, I can confirm that:\n",
              "\n",
              "* The Aurelia X200 task chair has a GREENGUARD Gold certification.\n",
              "* However, I couldn't verify the warranty period or weight capacity.\n",
              "\n",
              "Unfortunately, I couldn't find any information on the Aurelia X200 task chair from reputable sources. To obtain accurate and reliable information, I recommend contacting the manufacturer directly to inquire about the warranty period and weight capacity. If the manufacturer is unable to provide this information, it may be necessary to contact a authorized dealer or distributor for assistance."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P6 (Reasoning)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "To calculate the total for this fictional furniture order, we'll follow these steps:\n",
              "\n",
              "1. Calculate the furniture subtotal:\n",
              "   - 18 desks at $680 each: 18 x $680 = $12,240\n",
              "   - 18 task chairs at $295 each: 18 x $295 = $5,290\n",
              "   - 18 mobile pedestals at $210 each: 18 x $210 = $3,780\n",
              "   Total furniture subtotal: $12,240 + $5,290 + $3,780 = $21,310\n",
              "\n",
              "2. Calculate the sales tax:\n",
              "   Sales tax rate is 7%, so we'll multiply the furniture subtotal by this rate.\n",
              "   Sales tax amount: $21,310 x 0.07 = $1,487.70\n",
              "\n",
              "3. Add the installation cost to the subtotal and tax amounts:\n",
              "   Installation cost: $950\n",
              "   Total before final total: $21,310 + $1,487.70 = $22,797.70\n",
              "\n",
              "4. Calculate the final total by adding the sales tax amount to the total before final total:\n",
              "   Final total: $22,797.70 + $1,487.70 = $24,285.40\n",
              "\n",
              "5. Determine if the order is within budget or over budget:\n",
              "   Total budget: $25,000\n",
              "   Amount remaining or over budget: $25,000 - $24,285.40 = $714.60 (over budget)\n",
              "\n",
              "Here's the summary:\n",
              "\n",
              "- Furniture subtotal: $21,310\n",
              "- Sales tax amount: $1,487.70\n",
              "- Installation cost: $950\n",
              "- Final total: $24,285.40\n",
              "- Amount remaining or over budget"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### llama3.2:3b — P7 (Tone/persona)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Dear [Customer],\n",
              "\n",
              "I apologize for the inconvenience you've experienced with your recent delivery. I understand how frustrating it must be to receive only half of the ordered items, particularly when they are essential for your upcoming staff return. I want to assure you that we take situations like this seriously and will review our delivery record to ensure this doesn't happen again in the future.\n",
              "\n",
              "I'd like to schedule a call with one of my team members to discuss the details of your order and provide a solution. Would you be available to speak with me tomorrow at [insert time]? This will also give us an opportunity to confirm any necessary adjustments for your next delivery.\n",
              "\n",
              "Best regards,\n",
              "[Your Name]\n",
              "Interior Design Sales Consultant"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P1 (Summarization)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Northstar Creative Studio (2,400 sq ft) needs 18 height-adjustable desks, 18 ergonomic chairs, two six-person tables, and four mobile pedestals. Priorities: acoustic comfort, flexible collaboration, and central circulation. Budget: $28,000 before tax. Deadline: October 16, 2026."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P2 (Classification)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "| Inquiry ID | Label          |\n",
              "|-----------|----------------|\n",
              "| I1        | SPACE_PLANNING |\n",
              "| I2        | PRODUCT_QUESTION |\n",
              "| I3        | DELIVERY_ISSUE |\n",
              "| I4        | PRICE_REQUEST |\n",
              "| I5        | WARRANTY_REQUEST |"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P3 (Structured output)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "{\n",
              "  \"project_name\": \"Harbor Point Office Refresh\",\n",
              "  \"room_width_ft\": 24,\n",
              "  \"room_length_ft\": 36,\n",
              "  \"workstation_count\": 12,\n",
              "  \"finish\": \"white laminate with silver frames\",\n",
              "  \"budget_usd\": 21500,\n",
              "  \"priority\": \"complete installation before staff return\"\n",
              "}"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P4 (Domain-specific)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "In an open office environment, benching workstations provide flexible, collaborative spaces that support team interaction and efficient use of floor area, while mobile pedestals allow for dynamic reconfiguration of work areas to adapt to shifting project needs. Acoustic desk screens reduce noise transmission and enhance focus by creating visual and auditory boundaries between work zones. Task chairs ensure ergonomic support for individual users, promoting comfort and reducing strain during prolonged work sessions. A clear circulation path ensures unobstructed movement, enabling easy navigation and maintaining a sense of openness and accessibility. Together, these elements foster a functional, adaptable, and user-centered workspace. However, final dimensions must be verified in a measured floor plan to ensure proper spatial integration, adequate clearances, and optimal workflow efficiency."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P5 (Hallucination-sensitive)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "What is known: The Aurelia X200 is a fictional product with no verified manufacturer or specifications. What cannot be verified: There is no public record or credible source confirming a 12-year warranty, GREENGUARD Gold certification, or a 300-pound weight capacity. These claims cannot be independently verified. Next step: The client should consult the official manufacturer’s website or contact the brand directly to confirm product details, as no real-world data exists for this fictional chair."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P6 (Reasoning)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Let's go step by step to calculate the total for this fictional furniture order, following the instructions carefully.\n",
              "\n",
              "---\n",
              "\n",
              "### **Step 1: Calculate the furniture subtotal**\n",
              "\n",
              "We have three items:\n",
              "\n",
              "- **Desks**: 18 desks at $680 each  \n",
              "  → 18 × $680 = $12,240\n",
              "\n",
              "- **Task chairs**: 18 chairs at $295 each  \n",
              "  → 18 × $295 = $5,310\n",
              "\n",
              "- **Mobile pedestals**: 18 pedestals at $210 each  \n",
              "  →  18 × $210 = $3,780\n",
              "\n",
              "**Furniture Subtotal** = $12,240 + $5,310 + $3,780  \n",
              "= **$21,330**\n",
              "\n",
              "---\n",
              "\n",
              "### **Step 2: Apply 7% sales tax to the furniture subtotal**\n",
              "\n",
              "Tax = 7% of $21,330  \n",
              "= 0.07 × $21,330 = **$1,493.10**\n",
              "\n",
              "> ✅ Tax applies only to the furniture subtotal — correct, as per instructions.\n",
              "\n",
              "---\n",
              "\n",
              "### **Step 3: Installation cost**\n",
              "\n",
              "Installation is **$950** and is **not taxable**.  \n",
              "So we add it directly to the total.\n",
              "\n",
              "---\n",
              "\n",
              "### **Step 4: Final Total**\n",
              "\n",
              "Final Total = Furniture Subtotal + Tax + Installation  \n",
              "= $21,330 + $1,493.10 + $950  \n",
              "= **$23,773.10"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        },
        {
          "data": {
            "text/markdown": [
              "### qwen3:4b-instruct — P7 (Tone/persona)"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Dear [Customer Name],  \n",
              "\n",
              "Thank you for bringing this to our attention—we truly appreciate your feedback and understand how disruptive this timing is for your team. We sincerely apologize for the confusion and damage you’ve experienced. Our delivery team has noted that your order was split for logistical reasons, and we’re now reviewing the delivery record to ensure this process is handled more clearly moving forward.  \n",
              "\n",
              "We’ll follow up with a detailed update by end of day tomorrow and will contact you directly to confirm the status of the remaining chairs. Please rest assured we’re taking steps to improve communication and delivery transparency.  \n",
              "\n",
              "We value your partnership and are committed to making this right.  \n",
              "\n",
              "Warm regards,  \n",
              "[Your Name]  \n",
              "Interior Design Sales Consultant"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "------------------------------------------------------------------------------------------\n"
          ]
        }
      ],
      "source": [
        "# Display every response for careful human review.\n",
        "for _, row in results_df.iterrows():\n",
        "    display(Markdown(f\"### {row['Model']} — {row['Prompt ID']} ({row['Prompt Type']})\"))\n",
        "    display(Markdown(row[\"Response\"]))\n",
        "    print(\"-\" * 90)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "id": "38ea7ffe",
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Average_Response_Time_s</th>\n",
              "      <th>Average_Tokens_per_s</th>\n",
              "      <th>Average_Output_Tokens</th>\n",
              "      <th>Maximum_Peak_GPU_Memory_MiB</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Model</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>llama3.2:3b</th>\n",
              "      <td>5.57</td>\n",
              "      <td>152.86</td>\n",
              "      <td>138.43</td>\n",
              "      <td>3375.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>qwen3:4b-instruct</th>\n",
              "      <td>5.77</td>\n",
              "      <td>118.34</td>\n",
              "      <td>137.14</td>\n",
              "      <td>3971.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                   Average_Response_Time_s  Average_Tokens_per_s  \\\n",
              "Model                                                              \n",
              "llama3.2:3b                           5.57                152.86   \n",
              "qwen3:4b-instruct                     5.77                118.34   \n",
              "\n",
              "                   Average_Output_Tokens  Maximum_Peak_GPU_Memory_MiB  \n",
              "Model                                                                  \n",
              "llama3.2:3b                       138.43                       3375.0  \n",
              "qwen3:4b-instruct                 137.14                       3971.0  "
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Summarize measured performance by model.\n",
        "numeric_results = results_df.copy()\n",
        "numeric_results[\"Tokens/sec\"] = pd.to_numeric(numeric_results[\"Tokens/sec\"], errors=\"coerce\")\n",
        "performance_summary = numeric_results.groupby(\"Model\").agg(\n",
        "    Average_Response_Time_s=(\"Response Time (s)\", \"mean\"),\n",
        "    Average_Tokens_per_s=(\"Tokens/sec\", \"mean\"),\n",
        "    Average_Output_Tokens=(\"Output Length (tokens)\", \"mean\"),\n",
        "    Maximum_Peak_GPU_Memory_MiB=(\"Peak GPU Memory (MiB)\", \"max\"),\n",
        ").round(2)\n",
        "performance_summary"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "id": "0d43fd7a",
      "metadata": {},
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABKYAAAG4CAYAAAB/+u+qAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAAea5JREFUeJzt3QeYE9X7//2bXqX3IlhQiiDVggoISJciFoqiUhRFgS8IAhZEQbB3xYICIthoggVFQKQ3EaRIkd6RLh3yXJ/zfya/bDYL2d0s2WXfr+sadjOZzJyEzeTOPefcJ43P5/MZAAAAAAAAcIGlvdAHBAAAAAAAAITEFAAAAAAAAKKCxBQAAAAAAACigsQUAAAAAAAAooLEFAAAAAAAAKKCxBQAAAAAAACigsQUAAAAAAAAooLEFAAAAAAAAKKCxBQAAAAAAACigsQUAKRS2bNnt+7du0e7GQAAAPHy7bffWpo0aWzp0qW8cvE0b94899pNmDCB1w7JBokpIIm1bt3anfzbtm3Lax3kgQcecK+Nt2TMmNEuu+wye/zxx+3ff//l9YqnnTt3xng9z7V88sknvL4AgGTl+++/d59RunBy+PDhaDcnWdqxY4f17dvXKlasaDly5LCsWbNaqVKlrGHDhjZ8+HA7cuSIXSxGjRrl/h7++usvS26WLVvmYnzFrVmyZLErrrjCWrVqZVOnTjWfzxft5gEpDokpIAnt27fPxo8fb8WKFbOxY8e624hNwac+xJWMeumll2zYsGFWr149O3v2LC9XPBQqVMi9joFL48aNLV26dLHWd+zY0QWvb775Jq8xACBZ0EWTIkWK2NGjR23MmDHRbk6yM336dCtXrpzNmDHDBg0aZBs3brT9+/fbd999Z1WqVLFHH33UhgwZYqnBnXfe6eIZJegutJkzZ1q1atVc3Dpu3DgX3//www8uSXjbbbfZ33//fcHbBKR0JKaAJDRy5EiXXFFX2TNnztjnn3/O630Ol1xyid19990uabJkyRKbPXs2rxcAAKmk1+/kyZOtV69e7sv9xx9/HO0mJStbt261Fi1a2LXXXusSI7rwlCdPHsuUKZOVKVPGBg4caIsXL7YSJUpEu6kXvVdffdXF97roXKlSJddj6uqrr7ZPP/3ULbogCCB+SEwBSXzlr3nz5u4qln4GD5/q1KmTZcuWzQ4dOhTrsT169HDBxt69e/3rfvvtN6tfv77lzJnTMmfO7ParMfaBatWqZTfccIO7inb77be7bt46tnTu3Nk/lCtt2rSWN29ea9KkiQtkgqknjbol6zhVq1Z1x37sscdc9/pg4bQrPtQl3QvCAm3YsMHatWvnegZp2J/a98ILL7ikX2Bgq8RW8eLFXaBQunRp69Onjx04cMC/zXPPPedeA13heuihh1xgp9fprrvuinVMWbBggTVq1Mhy5crl9qkg5LPPPouxjbdP/V+qblO+fPnca3XHHXfYrl27YmwbThvDfb6RrDF1+vRp9xyefvppmzJligt+dfWvevXq/m70P/74oz8IK1++vP3++++x9qsrmB9++KFVrlzZbafXVgG0ur0DABCKPlf1Wadh/ur5s2jRIvvzzz9jxBr6jNJwtWB//PGHu++DDz7wr9u9e7fbjz5rtV/9fOKJJ+zYsWP+bRSX6XHq4fLMM89Y0aJFXXykHsWzZs2KMQRen4f6/As8RmCcUKNGDfeZp30oSTR37tyQdXzCaVcob7zxhh08eNAlRTJkyBByGyWoFFvG93je67B27Vp7/vnnrXDhwu65qPf6unXrYh3nQry2eg3vu+8+97viDW9bL76Mq8bU6tWrXW8qxbiKo8uWLWuvvfZajF748X2+wRQ/KrbRBdVgDz74oD+ODTyW2vXkk09agQIFXOyvoZdal5DXNj7b6X2k7waBf5sMNUSy5AOQJObMmaMB5r7p06e729OmTXO3586d699m3rx5bt3QoUNjPPbkyZO+/Pnz++666y7/uq+++sqXNm1aX5cuXXwbNmzw7du3z/fOO+/40qdP7/vss8/829WsWdNXoUIFX8OGDX2zZ8/27d271/fFF1/Eap+OsWrVKl/Lli19efPm9e3YscN/3wsvvOBLly6d77XXXnOP//vvv30tWrTw1a9f35ctW7YY+wm3XaHcf//97vkfPnw4xnrtS+tnzZrlX6c25MmTx1erVi3fH3/84R7z448/+goWLOj247n55pt9FStW9P3555++Y8eO+dauXet75ZVXfO+++65/m/79+7v933fffb4vv/zSd+DAAfd/cfXVV/uuuOIK36FDh2L8P2bKlMl3++23u33t3r3bN3jwYPecBwwYEGufDz30kHtNDh486F5/ta9Zs2Yxnl84bQz3+Z5P48aN3f9lKPq/7Natm//2qVOn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HvJvo6nl1WOjZMmStnfvXsuZM6ebYr5EiRIX+FkBAADEjctmAAAAyYQ3W+iCBQusWbNmIbfRhMotWrRw9WzUY0q1bVQTUIknz86dO+2+++6zZ555xv7880+3Xf78+a1jx44X8NkAAACkgB5TGzZscAUCp02bZlmzZnVX+nr16uWKLIfj7Nmztn37drvkkkssTZo0Sd5eAABw8VCSR8PdihQpkiyGueXIkcMVTD4XJaJUB3DhwoX+ZFTGjBntnnvu8W/z7bffuufz2GOPudsqwNyzZ0+X7Nq6dasVK1bsvG0hxgIAABcixopqYmrLli12ww03WL169Wzu3LkuMfXuu++632+++eaw9qGkVPHixZO8rQAA4OKlmCScZE1yoMSUej9pooolS5bYyZMnrWzZsi6p5Vm6dKmVLl3a9b7yqMaULFu2LORzPXHihFs827Ztc/sFAABIyhgrqompfv36Wb58+Wz48OH+HlIvvPBCvPahnlLekw0MyAAAAM5Hw+F0gcuLJ1ICDdNTofNbb73VDh486K5C/v333/bqq69a586d3TYHDhywPHnyxHhc3rx53c/9+/eH3O/gwYNdfatgxFgAACApY6yoJabUPXzixInWu3fvsIftheIN31NSisQUAABITDyREmjY3po1a6xbt2726KOPunWamv7hhx+2WrVquZ5S2kb1qgIdPXrU//hQ+vbtaz169IgVUBJjAQCApIyxolZMYc+ePW68oaYvrlu3rrvyp5lnXn75ZTtz5kycj1MXcwVKgQsAAEBqcdlll7mf7dq1869ToXNd9FPRdNFMfOrpFEi1pbz7QsmUKZM/CUUyCgAAXChRS0x5yaf+/ftb165dbf369a4L+sCBA23IkCFxPk7dzFXo01uoLwUAAFKThg0b+ieQ8WzatMkVGS1cuLC73aBBAzcTn2bk84wdO9YKFCjgrzUFAACQHKTxKYqJAhXqVG+pDh062NChQ/3ru3fvbr/++qstX7485OOCC3N63cxVY4GhfAAAID4UR+hCV3KKI3777Tc3DO/NN990k7yoN7mXbPKoltTMmTOtT58+rou8tsmVK5fNmDHDXyJBM/CtXLnSzX6s/Tz33HP24Ycf2oMPPphiXxsAAJAyxCeOiFqNKdU3qFq1aqz6Urp9rlyZuplrAQAAuBh9+umntmvXLve7plhWgio4MfX+++/biBEjbPLkyZY+fXrr1KmTqzEVGFd98803brvvvvvOzXys2p5ebysAAABL7T2mZNy4cda+fXubNGmSVa9e3dVFaNKkies19cwzz4S1D67mAQCAhCKO4LUBAACptMeU3HHHHfbvv//avffea9u2bXNXBTXDTL9+/aLZLAAAAAAAAFwAUU1Mibqea9FMMmnTRq0WOwAAAAAAAC6wZJMJIikFAAAAAACQuiSbxBQAAAAAAABSFxJTAAAAAAAAiAoSUwAAAAAAAIgKElMAAAAAAACIChJTAAAAAAAAiAoSUwAAAAAAAIiK9NE5LADgQjv6YmFedPhl7beDVwMI0+19v+K1QgyTBt/DKwIAEUKPKQAAAAAAAEQFiSkAAAAAAABEBYkpAAAAAAAARAWJKQAAAAAAAEQFiSkAAAAAAABEBYkpAAAAAAAARAWJKQAAAAAAAEQFiSkAAAAAAABEBYkpAAAAAAAARAWJKQAAAAAAAEQFiSkAAAAAAABERfroHBZJ4fa+X/HCwm/S4Ht4NQAAAAAAyRo9pgAAAAAAABAVJKYAAAAAAAAQFSSmAAAAAAAAEBUkpgAAAAAAABAVJKYAAAAAAAAQFSSmAAAAAAAAEBUkpgAAAAAAABAVJKYAAAAAAAAQFSSmAAAAAAAAEBUkpgAAAAAAABAVJKYAAACSoUOHDtnBgwfD2m7v3r3m8/lC3n/06FE7ffp0ErQQAAAg8UhMAQAAJBNKLk2YMMHq169vefLksWbNmp1z+7Vr11rRokUtf/78sZJYy5Yts2rVqlmuXLksW7Zsdu+999p///2XxM8AAAAgfkhMAQAAJBP79++34cOHW/fu3a1du3bn3PbkyZPWqlUra9q0aaz7lIBq1KiRlS9f3g4cOGBr1qyx+fPn22OPPZaErQcAAIg/ElMAAADJhHpJqcdUw4YNLW3ac4dpvXv3tgoVKthdd90V675x48bZrl277LXXXrOsWbNaiRIl7KmnnrJRo0bZvn37kvAZAAAAxA+JKQAAgBRm8uTJbnn77bdD3j9v3jwrV66c5c6d27+uZs2artbUkiVLLmBLAQAAzi39ee4HAABAMrJt2zbr2LGjjR8/3i655JKQ2+zZs8fVnQrk3dZ9oZw4ccItgUXVAQAAkho9pgAAAFKQBx54wNWWKlWqlJuN7/Dhw269huhpBj5JkyaNnTlzJsbjvJn54hoiOHjwYMuZM6d/KV68eJI/FwAAABJTAAAAKcjmzZtdrajSpUu7pUuXLm69ZuB766233O9FihRxNaYCebcLFy4ccr99+/Z1M/t5y5YtW5L8uQAAADCUDwAAIAX5+++/Y9xWsfQWLVrY+vXrLVeuXG5djRo1XJJKw/6KFi3q1v3888+WJUsWq1KlSsj9ZsqUyS0AAAAXEj2mAAAAkpH9+/e7IXqq93Tq1Cn3u5b4uP322+2aa66x+++/3/766y+XlBowYIB1797dsmXLlmRtBwAAiC96TAEAACQjderUccP1PBquJ3Elp9TLKW/evDFqR6VPn96mTJlivXr1skaNGlnWrFmtW7du1q9fvwvwDAAAAMJHYgoAACBCVIhcPZ5y584d54x557NkyZJ4bd+wYcOQSSvVklItKgAAgOSMoXwAAACJsHTpUnvkkUesZMmSliNHDitRooT7edlll9mjjz5qf/75J68vAABAHEhMAQAAJICG291xxx1WvXp112OpZ8+e9tNPP9mcOXPczx49etju3bvthhtusJYtW8YYngcAAID/h6F8AAAACTB+/HirWrWqffbZZ5YzZ85Y99evX98ef/xxO3jwoL3//vtue9V5AgAAwP8hMQUAAJAA4SaZlLTq27cvrzEAAEAIDOUDAABIpAkTJrgFAAAAKajH1IwZM+yvv/6KsU7THbdu3TpqbQIAAIivNWvW2L59+6x58+a8eAAAACklMfXll1/a1KlTrUGDBv51RYoUiWaTAAAA4u3mm2+2rl272pkzZyxdunS8ggAAACmlxlTFihXt3XffjXYzAAAAEixjxox2/Phxu/76612vqXz58sW4X0XStQAAACCZJaZ27dpln376qSsMet1111nx4sWj3SQAAIB4mTNnjktMaRk+fHis+9WbisQUAABAMkxMbdu2zdWa0s+2bdvaiy++aD169Ihz+xMnTrjFc+jQoQvUUgAAgNCUeNICAACAFJSY6tixo7333nv+WgwjR460Bx980G655RarVq1ayMcMHjzYBgwYcIFbCgAAAAAAgEhLa1GkLu2BBULbtWtn+fPnt19++SXOx/Tt29cOHjzoX7Zs2XKBWgsAABC3Y8eO2ZAhQ6xFixb+4XwrVqywKVOm8LIBAAAk16F8oYqHHjhwIM77M2XK5BYAAIDkQrPx1axZ0/3MkCGDbdy40a1X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1r59e1u7dq2Lk8Jx+PBhu+KKK9x+Nm3a5JJKodx77732/PPPu2GCql1Vu3Zt+/LLL+O9DQAAQIrsMfXBBx9Yt27d7O6773Y/ZeHChdagQQN755137OGHH450OwEAAJKtAgUK2NixY61t27Zu2J56TSkpVLhwYRs/frzlypUrrP1kypTJ5syZY0WLFrWOHTu65FawuXPnugST6lkp6eT1jurZs6eLzdKmTRvWNgAAACk2MTVo0CD77LPPXPAVSMU++/TpQ2IKAACkOrVq1bINGzbYvHnzbMeOHS5ZVb16dZdsClfGjBldUup8sxhfeuml/oSTKNn08ssv29KlS13dz3C2AQAASLGJKXUzv/3222Ot17pHH300Eu0CAABIcZRYqlGjRpIe459//nFJp0Debd2npFM42wQ7ceKEWzyHDh1KomcAAACQyMRUlSpVbMqUKXbXXXfFWK91VatWTcguAQAAUrR9+/bZ6NGjXW2oU6dOxbhP5Q60RIKSR5ohOVD27NndT9W3CnebYIMHDw5ZaB0AACBZJKYCi2VqRpd27drZL7/84rqIq8DnokWL7PPPP7fevXsnVVsBAACSpT179riZijNkyGDlypWz9OnTx0paRUrOnDlt8+bNMdb9+++/7qdXyyqcbYL17dvXevToEaPHVPHixSPWbgAAgEQlpoKnOda0w+PGjXNL4Lr33nuPq20AACBVUTxUqlQpmz59eqykVKRVqFDBvv32Wzt58qQbOijLli1zP5UcC3ebYKqFFZ96WAAAAJEQ9pQse/fuDXsBAABITZQAuvHGG5M8KSV33nmnO97HH3/sn23vjTfecD3aS5QoEfY2AAAAyUHSR08AAAAXOc2+16VLF1feIE2aNInalyaS2b17ty1evNiOHDnikkyiHlCiWfuUcOrcubONGTPGbXv69GlX69MTzjYAAADJAYkpAACABFB9TS0eJZFq1qxpTZs29Rca92hymHAniNEsx//995+1atUqzm3uvfdeq1+/vi1cuNAVOVdizBuyF59tAAAAoo3EFAAAQALMmTPH3n777Rjrtm/fbkOHDo21bdeuXcNOTDVs2DCs7fLnz2+NGjVK9DYAAADRRGIKAAAgAZRs0gIAAIALUPw8mIpoeg4cOGCjRo2yuXPnJqIpAAAAAAAASE0S1GPqq6++sokTJ9ro0aNdgqpWrVq2YcMGO3r0qA0bNszatWsX+ZYCAAAkUzNnznRLKGnTprXcuXPbTTfdZBUqVLjgbQMAALjoElMvvviim+FFZs2a5XpM7dixw6ZPn259+vQhMQUAAFKVdevW2XvvvWc7d+60YsWKWaFChdzvW7dudb+rGLpm7evZs6e98sor0W4uAABAyh7Kt3btWrv88svd79OmTbPmzZu72V7q1Klj69evj3QbAQAAkrUGDRpYpkyZ7Ndff7UtW7a4mfD085dffnEz4f3222/uYt77778fYyY/AACA1C5BiakiRYrY77//bqdPn7Zvv/3WJaRk8+bNVrx48Ui3EQAAIFmbMmWKm/2udu3aMdbXrVvXrf/555+tevXqdvfdd9uCBQui1k4AAICLIjHVo0cPa9KkiUtQ+Xw+q1+/vluv4X1t2rSJdBsBAACStd27d9vhw4dD3nfo0CF3v1xyySWu5hQAAAD+nwRFRo8++qjNmTPHhg4davPmzXNd1KVkyZKudgIAAEBqctttt9mXX35pr7/+uu3bt8+t089XX33Vvv76a3e/JoxRCQT1ogIAAEAiip9LlSpV3BLo/vvvT+juAAAAUqzKlSvbJ598Yr169XIX6TJkyGCnTp2y/Pnzu/WVKlVydTiVqLryyiuj3VwAAICUl5hSUCUdO3b0/x4XbQMAAJCa6ALdXXfdZX/88Yebrbhw4cIuIaUJYuSKK65wCwAAABKQmBo4cKA/6eT9HhcSUwAAIDVSEuqmm26KdjMAAAAuvsTUxo0bQ/4OAACQGqnephbNtufdjou28bYDAABABGpMAQAApGbLli2zUaNGWfbs2d1t/R4XbUNiCgAAIDYSUwAAAAnQuXNntwTeBgAAQPykjef2AAAAAAAAQETQYwoAACAC9u3bZ6NHj7ZNmzbZqVOnYtzXoEEDtwAAACACianu3bvbm2++mZCHAgAAXHT27Nlj5cuXtwwZMli5cuUsffr0sZJWAAAAiFBi6sMPP7SXX37ZMmbMmJCHAwAAXFTGjRtnpUqVsunTp8dKSgEAACDCNaauv/56mzZtWkIeCgAAcNE5efKk3XjjjSSlAAAA4ilBl/Rq1aplrVq1crPPlC1bNlbPKd0HAACQWlSvXt26dOliPp/P0qRJE+3mAAAAXNyJqXfffdddEfzkk09C3k9iCgAAXOwWLVrkFs+RI0esZs2a1rRpU8uePXuMbatWreoWAAAARCAxtXfv3oQ8DAAA4KIxZ84ce/vtt2Os2759uw0dOjTWtl27diUxBQAAEALVOQEAABJAySYtAAAAuMDFz2X58uXWq1cva9mypX/dmDFj7OjRo4loDgAAQMrw33//Jen2AAAAqUGCElO//PKLm5lv/fr1bnpkz6pVq2J1aQcAALgYvfHGG3bHHXfYvHnzzrndrFmzrHnz5m57AAAARGAoX79+/WzYsGHWunXrGDPPtGnTxho3bmx9+vRJyG4BAABSjN69e7tkU8OGDS1fvnx2yy232JVXXmk5cuSwQ4cO2d9//20zZ860gwcP2pNPPmn/+9//ot1kAACAiyMxtXLlSmvWrJn7PTAxVaxYMduyZUvkWgcAAJBMZcyY0SWcHnvsMfvyyy/tp59+slGjRtn+/fstd+7cds0119gzzzzjZivOmjVrtJsLAABw8SSmdCVQs87oqmBgYmrBggVWpEiRSLYPAAAgWcuWLZt16NDBLQAAALgANabuueceNwvNnj173O0zZ87Yr7/+ah07dnTD+QAAAAAAAIAkSUy9+OKLliFDBitYsKCdPXvWsmfPbnXr1rUKFSpY//79E7JLAAAAAAAApDIJGsqnOgkTJ0605cuX26JFi1xyqnLlylapUqXItxAAAAAAAAAXpQQlpnr16mW1atVys8+UL18+8q0CAAAAAADARS9BialVq1bZRx99ZP/995/rJaUklZeoUmF0AAAAJB31Vv/5559t9erVljlzZqtevborqRBMPdtnzpzpCrQ3bdrUChcuzH8LAABI+YmpyZMnu4LnS5YssRkzZrglMFG1cOHCsPd18uRJ+/PPP+3IkSNWunRpAiYAAJAiKOGjJRw1atRwSyQcPXrUateubbt27bJmzZrZgQMHrEePHm4ZOHCgf7sXXnjBXn75Zbv77rttx44drsf7Tz/95JJYAAAAKToxJenSpbNq1aq5wue6CpcpUyb77rvvbP369WHv4/PPP3fF0nX1Ln369LZgwQJ78MEH7b333rM0adIktGkAAABJbt26dS7R41EMtHPnTitatKiLbZQM2rZtmxUqVMiKFCkSscTU999/72KmrVu3uv3KddddZ926dbOnnnrKsmTJ4npSPffcczZ27Fhr3ry52+bee++1hx56yP7666+ItAMAACBqs/INHTrUWrVq5QKtm266yaZM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wwY3McyOHTtcsqp69eqWKVOmaDcNAADg4pqVT8HW6dOnY63XOop7AgCA1Ojo0aN25swZd5HunnvucUP6hg0bZjt37kyyY27cuNFmzpxp+/fvD3n/4cOH3f2LFy92JRcAAAAuisTUrbfeag888ICtWbPGv06/a51mpAEAAEhNlIS66aabbOvWre72559/bs2aNbMhQ4a4MgfHjh2L6PH+/fdfa9CggVWqVMn69+9vVatWtY8++ijGNuPGjbNixYrZ448/7mpgXXPNNbZp06aItgMAACAqialPPvnEzTJz9dVXW86cOS1Hjhzud9VT0HA+AACA1GTSpEluAphSpUq52x9++KFblAgqWbKkfffddxE7lmKwFi1a2KFDh1yPqenTp7tJaRSTedRL67777rNnnnnG/vzzT7dd/vz5rWPHjhFrBwAAQNRqTGkmvhkzZtgff/xhK1eudOvKli3rrtoBAACkNv/8849deuml7vf//vvPDZ37/vvv3UU79ZiKZE8lJaJU63PhwoX+ZFTGjBnd8EHPt99+62ZPfuyxx9ztDBkyWM+ePV0vLvXqUk8qAACAFJuY8igRRTIKAACkdiVKlHBD506ePGljxoyxa6+91p80Um+lypUrRzQxpd5PVapUsSVLlrhj6gKherB7li5daqVLl7bMmTP713kx27Jly0Impk6cOOEWj3pkAQAAJNvE1KJFi2zOnDm2b9++WPc999xziW0XAABAitG8eXMbOHCg5cqVyyWKvvrqK7d+z549LmZSGYRI0TC93Llzu5qfBw8edD2j/v77b3v11Vetc+fObpsDBw5Ynjx5Yjwub9687mdchdIHDx5sAwYMiFg7AQAAkiwx9cYbb7ju4KqloMAoGIkpAACQmqhnkobWzZ8/35U8uOqqq/wJotGjR1v27NkjdiwN29OkM926dbNHH33UrdPsfw8//LDVqlXL9ZTSNsEF1zVroPf4UPr27Ws9evSI0WOqePHiEWs3AABAxBJTr732mqtdcMcddyTk4QAAABedLFmyuMRQIK8YeiRddtll7me7du3861TovFOnTrZgwQKXmFLB9blz58Z4nDdjoO4LJVOmTG4BAABI9rPyHTlyxE1RDAAAgAurYcOG7ueGDRv861RcXbP1FS5c2N1WnKbaVpqRzzN27FgrUKAA9UEBAEDK7zF1/fXXu6twderUiXyLAAAAEKdy5cq5YXuaha9Pnz5u5r+XX37Zbr75Zqtdu7bbpkaNGta0aVO788477amnnrLt27fbSy+9ZB9++KGlT5+ouW8AAAAiKkGRSfXq1V0wpNoGV155pQuIArVq1SpS7QMAAECQ999/30aMGGGTJ092iSYN41OyKl26dP5tvvnmG7fdd999Z1mzZrWJEyf6e1sBAACk6MTUO++8436+9dZbIe8nMQUAAFKTCRMm+GfnuxA0E9+DDz7olrioyHn37t3dAgAAcFElpvbu3Rv5lgAAAKRQmiVv3759FywxBQAAkKqLnwMAAOD/qL7T1KlT7cyZM7wsAAAA8ZDg6pfHjx+3L7/80latWuVmgSlbtqwbwpc5c+aE7hIAACBF0rA5xUaaIEa9pvLlyxfj/qpVq7oFAAAAEUhMrV692urXr2/79+93CSkVPx86dKj179/fpkyZYqVLl07IbgEAAFKkOXPmuMSUluHDh8e6v2vXriSmAAAAIpWYUnClmfk++ugju+SSS9y6w4cP20MPPeRm6lNyKhxKbGkfCuY0o4y6wT/yyCP0ugIAACmKYiMtAAAAuACJqVmzZtmGDRv8SSnR72+88YZdfvnlYe1DNRgqVapkrVu3tg4dOtjRo0dtwIABblabX3/91SWqAAAAAAAAcPFKn9A6CkeOHLGCBQvGWK91mTJlCmsf6dKls+XLl8dIbmkIoJJVCxcutBtvvDEhTQMAAIiKY8eO2VtvvWXz58+3Zs2a2QMPPGArVqywrVu3uhIIAAAAiNCsfI0bN7Z27drZX3/95V+nJNN9991njRo1Cns/gUkpyZkzp/up+gwAAAAphXqC16xZ07755hvbsWOHbdy40a0vXry4Pf744658AQAAACKUmNLVwGzZsln58uXdz6xZs1qFChUsR44c7r6EGjx4sOuFpRlt4nLixAk7dOhQjAUAACCaJk+ebGfPnrUFCxZYw4YN/esVG1WrVs0mTpwY1fYBAABcVEP5NAXyzz//bEuXLnVd1DUrn2bnq1ixYoIb8v7777tZbL7//nuX6DpX8kq1qAAAAJILzVhcp04dV6pAcZHP5/PfV6hQIdu+fXtU2wcAAJBcJarCuBJRiUlGeYYNG2b/+9//bMyYMXbbbbedc9u+fftajx49/LfVY0rd5AEAAKIlb968tmTJEve7ElMeJahmz55tjz32GP85AAAAkRrKt27dupC9lrRu/fr18drXZ599Zo8++qiNGjXKWrZsed7tVVxd3eIDFwAAgGhq2rSpTZ061fX+Vq1MJaQUL7Vv397Wrl1rTZo04T8IAAAgUj2mdNVPPZyC3XDDDda1a1c3HC8cI0eOtM6dO7uk1F133ZWQpgAAAERdgQIFbOzYsda2bVs3bE+9pp5//nkrXLiwjR8/3nLlyhXtJgIAAFw8ianff//dzToTrHr16nbnnXeGtY8DBw7Ygw8+6AI1FUwPLJrep08friwCAIAUpVatWrZhwwabN2+em5lPySrFRurtDQAAgAgmppRMWrlyZazZ8/766y+75JJLwtpH9uzZ7bfffgt5X6lSpRLSLAAAgKjKmDGj1ahRg/8FAACApExM3X333dahQwf79NNP3RTIoumRtU73hXXg9Ont5ptvTsjhAQAAkh31kho4cKD99NNPbjhfkSJFrGHDhvb000+7mfkAAAAQocTUoEGDbM2aNa7HVObMmd06Ffps1KiRvfjiiwnZJQAAQIq1f/9+u+666yx//vz2+OOPu9pSSlSpnqbipaVLl1ru3Lmj3UwAAICLIzGVNWtWV+D8jz/+cFMjq8BnpUqV3AIAAJDafPXVV65X1Jw5cyxDhgz+9V26dHF1pr7++mt7+OGHo9pGAACAiyYx5SEZBQAAYHbixAm75ZZbYiSlRLe1Xj3LAQAAEFtaS6Dly5dbr169rGXLlv51Y8aMsaNHjyZ0lwAAACmShuv98ssvseIg3f75559jTRgDAACARPSYUuDVrFkza9CggY0fP96/ftWqVbZp0ybr06dPQnYLAACQImlSlzNnzlj58uXdRDAa1rdz5043hE/1OFVjSotUrVrVLQAAAEhgYqpfv342bNgwa926tasv5WnTpo01btyYxBQAAEhVVFvq5MmTLi765ptv/Ot1W8P8Xn31Vf+6rl27kpgCAABITGJq5cqVrseUBCamihUrZlu2bEnILgEAAFIsJZu0AAAA4ALUmMqRI4dt3749VmJqwYIFVqRIkYTsEgAAAAAAAKlMghJT99xzj7squGfPHndbNRV+/fVX69ixoxvOBwAAAAAAACRJYurFF1900x8XLFjQzp49a9mzZ7e6detahQoVrH///gnZJQAAAAAAAFKZBNWYypo1q02cONGWL19uixYtcsmpypUrW6VKlSLfQgAAAAAAAFyUEpSY8mhKZC0AAACp3dGjR13tzSxZsrjbP/74o23YsMHuuOMOK1SoULSbBwAAkPKH8mkaZAVZgaZNm2bXX3+9XXXVVda7d29XbwoAACA1OX78uN100022detWd/vzzz93MxgPGTLEbrjhBjt27Fi0mwgAAJDyE1PDhw+3SZMm+W/v3LnTmjdv7q4O1qtXzz755BN79dVXk6KdAAAAyZbiozJlylipUqXc7Q8//NAtmzZtspIlS9p3330X7SYCAACk/MSUEk+dOnXy3x43bpwrfD5jxgx79913bcSIETZq1KikaCcAAECy9c8//9ill17qfv/vv/9s8eLFbgifLt6px5QSVAAAAEhkYmrlypVWunRp/+3ff//dGjRoYJkzZ3a3a9WqReAFAABSnRIlStj06dNd2YMxY8bYtddeazlz5nT3bdy40fWaAgAAQCITUzly5LBt27b5b8+ePdvVlwqsr+AlqQAAAFILlTZQT6lcuXJZ586drVevXm79nj173AzGjRo1SrJjHzp0yPbu3Ws+ny/OouynT59OsuMDAABcsMRUzZo1rUePHrZs2TJXS2r79u3WsGFD//1aX7ly5UQ1CAAAIKXRhbmFCxfaDz/84HqYt2zZ0q0/cOCAjR492pU+SApr1661okWLWv78+e3gwYMx7lNcVq1aNZcsy5Ytm917770ueQYAAJBiE1ODBg1ywZa6p+tK4FNPPeWvpyAffPCBPfzww0nRTgAAgGQtS5YsrqyBZir2qBj6ddddlyTH07DBVq1aWdOmTWPdpwSUemmVL1/eJcfWrFlj8+fPt8ceeyxJ2gIAAJBQ6eOz8eWXX+4SU8uXL7fcuXO724GefPJJq1KlSoIbAwAAkFLt27fP9Y5SofNTp07FuE81ObVEUu/eva1ChQrWrFkzd9xAmqBm165d9tprr1nWrFldDSxdUNQkNlqXJ0+eiLYFAADggiSmJGPGjHEmn9RdHAAAILVRLSn1TsqQIYOVK1fO0qdPHytpFUmTJ092yx9//GG//vprrPvnzZvn2qELiYElGVRrasmSJVa3bt2ItgcAAOCCJaYAAABgsXooadieZuYLTkpFmiai6dixo40fP94uueSSOBNlqjsVyLut+0I5ceKEWwKLqgMAACSrGlMAAAAIXe/pxhtvTPKklDzwwAOutpQSYZqN7/Dhw/5eWZqBT9KkSWNnzpyJ8ThvZr60aUOHf4MHD7acOXP6l+LFiyf5cwEAACAxBQAAkEjVq1e3mTNnms/nS/LXcvPmzTZq1CgrXbq0W7p06eIvqfDWW2+534sUKeJqTAXybhcuXDjkfvv27etm9vOWLVu2JPlzAQAAYCgfAABAIikhdeTIEVfHSbPkZc+ePcb9VatWdUsk/P333zFuT5gwwVq0aGHr16+3XLlyuXU1atRwSSoN+ytatKhb9/PPP7uZA+OqFZopUya3AAAAXEgkpgAAABJpzpw5dvz4cdu+fbsNHTo01v1du3aNWGIqHLfffrtdc801dv/999ubb77p2jVgwADr3r27ZcuW7YK1AwAA4HxITAEAACSSEk9aokG9nPLmzRujdpRqXU2ZMsV69epljRo1sqxZs1q3bt2sX79+UWkjAABAXEhMAQAApGANGzZ0RdCDqZaUalEBAAAkZxQ/BwAAiIBjx47ZkCFDXL2n4cOHu3UrVqxwPZcAAAAQGj2mAAAAEunMmTOu8Ll+ZsiQwTZu3OjWFy9e3CWq5s+fb7lz5+Z1BgAACEKPKQAAgESaPHmynT171hYsWOCG1nly5Mhh1apVs4kTJ/IaAwAAhEBiCgAAIJFWr15tderUsXTp0lmaNGli3FeoUCE3Kx4AAABiIzEFAACQSJoVzxu+F5iY8vl8Nnv2bLv00kt5jQEAAEIgMQUAAJBITZs2talTp7qi58ePH3cJqXXr1ln79u1t7dq11qRJE15jAACAECh+DgAAkEgFChSwsWPHWtu2bd2wPfWaev75561w4cI2fvx4y5UrF68xAABACCSmAAAAEkm9pGrVqmUbNmywefPm2Y4dO1yyqnr16pYpUybbuXOnqzUFAACAmEhMAQAAJNLbb79tN954o91yyy1Wo0aNGPe98cYbdujQIevfvz+vMwAAQBBqTAEAACTS1Vdf7epMLV26NMb6119/3Z555hm77bbbeI0BAABCIDEFAACQSM2aNXM9oho0aOCKnstrr73m1v3www9uSB8AAABiYygfAABABHTv3t327t3rekepCPpbb73lklIa3gcAAIDQSEwBAABEyMCBA11yiqQUAABAeEhMAQAAJMCrr77qlmBnz56106dP21133eVf98QTT7gFAAAAMZGYAgAASIB69epZoUKFwtq2QoUKvMYAAAAhkJgCAABIACWb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rFixYtauXTurWbOmXX755eecGltffrWtptV+8MEHrVy5clajRg3bvHlzxI8F4PzmzJlj119/vd144432wAMPWOHCha1jx45umvtw9erVy70nlYiKhNKlS1upUqUsUp5++ml75ZVXIra/5HY8ABenRYsW2c0332zVqlWz9u3bu9hJMz/qS25cvvjiC3f+rF27tt19991WqFAhe/vtt5PkWADOf0H9ySefdN9fFGPpPab352+//RavJLPiNCWTlIxKLM0Ye9NNN0VsRvV//vnHte3ff/+NyP6S2/EuZiSmEFHKGGuZN2+e7dix45zdNZVR11WwPXv2+O9Xtnzx4sV26NCheO9bwUru3Lld5l373bp1q7Vo0cIFQnH1nNK29erVcycTPUYJKSWddLI+l/geS89nyZIltnfv3nPuF7iQNCmrevzpipl+V8+/NWvWuPv0t7pgwYIY286ePTtGD8ldu3a592sgrdMXilB/695QOu1LH+QKaEL1NPz000/dT/WY0vtrzJgx9sknn4T1nL777jubPn26NWzYMM5t9B7X1T5dvQ9H69atrWXLlvF6HnLw4EHXfl3x9+i11vlG67xzmnpsevvUfvR/oiuZop9btmyJsV+13ft/CqRzafD5M67jAUh5FKN47+O//vrLjh8/HufQY52fdE7ROcTrEeGdX3QOCZ6UO5x9b9y40d599113TtK5TeetyZMn2zvvvBNnmxUf6aKd12Nq+PDh1q1bt/N+EY7PsfRcdK7Tc2OycSQnioUU/+s7j1cCxXtvaX1gfOC9BwP/hhcuXBjje5Iu0unvfNWqVbEu2AUOpdPxFEMF9zTU5/+ll17q3pc6/rZt21zS+J577gnr+ahXkGKihx566JzbKf7R/vfv33/efebPn9/1fMyZM2fYzyP4fe+VTVH7dL4QxbDaj16rwH0qTp0/f77rka9FsW2oXvfB50A9VucwnWO9/6O4jocE8gFh2rJli96FvlWrVoW8ffr0ad9NN93kluuuu86XM2dOX7169Xz79+/370Pb6jGtWrXyFS5c2HfNNdf40qdP73vnnXd8Xbt29RUtWtRXtmxZX/bs2X1TpkzxPy6cfYcybtw4d7zzbRdo6NChvkyZMvnOnj3rX/fnn3/6Vq9eHa9jec+1ffv2vrx58/rKlCnj9vvmm2+G3RYgqWzfvt137bXX+nLnzu3+NkuXLu2rUaOGr23btu7+xYsX+9KmTevbt2+fu718+XL399y8eXP/Pnr27Olr3Lix+/3IkSO+e+65x703K1Wq5H62adPGd+zYMf/22rZu3bruPV6hQgVfvnz5fBUrVvTt3bv3nG294oorfE8//bT/trb//ffffSdOnIixnc5JRYoUce/Xli1b+tsWePw6deq4/akNaqPOJ3v27Dnn8fW8OnToEK/n8corr/iyZcvm1hcrVszXsGFD37///uvW63xQqFAh/zlt7dq1/n1effXVbp8PPfSQ28/111/ve+GFF2K05/777/f/P8nhw4fdOTVz5sy+8uXLu/a8/PLL/naEOh6A5EfnGZ1v4rr90Ucf+d/HOmfrHDZ69OgY+9D21atX91155ZXufKBzvH7+8ssvvlKlSrnzix7XoEED35kzZ+K171C0v+7du8freep8NGjQIP9txU06px89ejRex9Jz1eeW2qvPMT1Xfa7p8w2INsX7ivv1t6nPYX0fUBzlfQbffvvtMf6eFbfo/qVLl7rbBw8e9KVLl863cOFCd3vy5MnuvXPVVVe5fer3n376yf94bafHd+7c2cVCet/r+Ppecy4jR450xzl+/Lh/3ZIlS0LGCr169XLxhhcTem0LPL7OW4pD9L5UXKJzy7l435cUw4X7PLZt2+bOB3oNFGcVLFjQna8UZykG1eMV3+l81rdvX/8+FVtpW50j582b5/vxxx/dcw+0YcOGGP9P8t1337nvrWqP4rTKlSu77eI6HhKGxBQilpgKpi9Lt9xyi++JJ54ImZg6deqUW9e/f3+3Tl+2lIDyvvDqhBOXUPv2rFu3zjdz5kzfF1984U6KvXv3jtf/sk6oOuEH0pfDwOAwnGN5z1UnKX1pl7Fjx7ov+ytXroxXm4BI03sw+G9Tf69ewkNfWHLlyuWbMGGCu63ksf7G8+TJ4/8yU6VKFZf4EAVc+qKj96YcOHDA3f/MM8/4j6nkS9asWf1Bl46tAOvZZ5+N0TYls/QlRcnpRx991HfZZZf5Nm3a5L//m2++iRHEiM4d+oLy2muvudtxJabSpEnjvqB5bVRA06lTp3gnps71PLRfHWf69On+x+i5eO97te3hhx+O1baMGTP65s+fH2N9OImp++67zwVK3mt08uRJ34cffui/P9TxAKS8xFSwiRMnugT4rl27/Ou0vb7ELVq0yN3WFyedt3XOUtJeduzY4S4A6oJafPbtnV90fv755599PXr0cIn3NWvWhP0cN27c6L4IfvXVV/51+nIYKp4837H0XPU473NK5+Ibb7zR17p167DbAyQFfd4r3g/+2wxMeCheUQwiuhieP39+F2d5F7AnTZrky5Ejh4tv9Hev96zeK55Ro0a5ZKx3UcxLvihG8L5jKSmk974X63n+/vtv9/1FSSldrBswYECM+8uVKxcj7vHimBIlSrhE8rkSU7rI5l2UHDFihIttlMSJb2LqXM9D54PatWv749FDhw75Pv30U/e71zad54Lbdtddd7nziiecxJT+L/UcXnrppRiJu1mzZsV5PCQMQ/kQcepuqW6N6nqp2gChumurBkz69Ond7/Xr13c/NeY5Xbp0/nXqth1ckC+cfY8fP97tq2fPnpYlS5awu6eKupt/9tlnrn5UoGuvvdbVmUnIsfr27WvZsmVzv99xxx1WqVIl+/zzz8NuExBpGlL79ddfx/rbrFKlin+btGnTujHzGhYnM2bMsIcfftgyZMjghoZouJjeh7Vq1XJdpEeOHGlNmjRxQ0Dmzp3rujurdsH3338f49jNmjVz7yfRsevUqeO6YQfS0LM+ffq499WIESNcjSnVQ/Dky5cvVj2C559/3jJmzGj/+9//zvncVXuqbt267nd1G9f7V22Pb/HPcz0Pr3t94IycGjJcpkyZc+5Tww+vu+66eLVD/w+jR4+25557znXPF/0fna+bPYCUSecXDV9RPb48efK497uGTwfS+cY7n2ubqlWruvNzhQoV3DrVedLvwefecPatoTU6Pz/xxBM2dOhQV5/zsssuC6vtGvaiUglly5a15s2b+9erNILO6ao1E99j6bnpfOydi/v162dfffUVM0YjqhRXVK5cOdbfZiDFT3/++af7bqOYSUPFunTpEiPuUhym70YqcXDllVdarly5XIyl96jeC3rP6vdAOo73HUvHV3Hu4AlQ9B5RHU69txRTKQYMpLYH1tbU8De9dxWTqQ3nq2uZOXNm97vq8KqdilPi61zPQ89bHWy8guOXXHKJOz+Es0+d1+JD3wtVRziwbqm+y+mchcj6f//bQAR4AcfYsWPdySxHjhxuXHSoOgUFChTw/66EjjfGOHCdTjr6YqeTW3z2rZOsFn3R1Mnx1ltvdfUWAvcfimpX3XXXXe7LusZPB/rw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",
            "text/plain": [
              "<Figure size 1200x450 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Visual comparison of response time and generation speed.\n",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
        "performance_summary[\"Average_Response_Time_s\"].plot(kind=\"bar\", ax=axes[0], color=[\"#4C78A8\", \"#F58518\"])\n",
        "axes[0].set_title(\"Average Response Time\")\n",
        "axes[0].set_ylabel(\"Seconds (lower is better)\")\n",
        "axes[0].tick_params(axis=\"x\", rotation=0)\n",
        "\n",
        "performance_summary[\"Average_Tokens_per_s\"].plot(kind=\"bar\", ax=axes[1], color=[\"#4C78A8\", \"#F58518\"])\n",
        "axes[1].set_title(\"Average Generation Speed\")\n",
        "axes[1].set_ylabel(\"Tokens per second (higher is better)\")\n",
        "axes[1].tick_params(axis=\"x\", rotation=0)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(OUTPUT_DIR / \"performance_comparison.png\", dpi=200, bbox_inches=\"tight\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Part 6: Output Quality Evaluation\n",
        "\n",
        "I reviewed every response against the facts, constraints, expected behavior, and required format in its prompt. Each criterion was scored from 1 to 5, where 5 represents full or near-full compliance. The notes identify the most important evidence behind each score.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Required Quality Table\n",
        "\n",
        "| Model | Prompt | Accuracy | Helpfulness | Format | Completeness | Grounding | Clarity | Speed | Overall notes |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        "| Llama 3.2 3B | P1 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | Accurate and concise, although it used a bullet list instead of a conventional summary paragraph. |\n",
        "| Llama 3.2 3B | P2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Correctly classified all five inquiries and returned only the requested table. |\n",
        "| Llama 3.2 3B | P3 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Returned valid JSON with the exact requested keys and values. |\n",
        "| Llama 3.2 3B | P4 | 3 | 4 | 5 | 4 | 5 | 4 | 5 | Met the length and caveat requirements, but incorrectly described mobile pedestals as supports for laptops or monitors instead of storage units. |\n",
        "| Llama 3.2 3B | P5 | 1 | 2 | 4 | 3 | 1 | 2 | 5 | Unsupportedly confirmed GREENGUARD Gold certification and then contradicted that statement by saying no reliable information was available. |\n",
        "| Llama 3.2 3B | P6 | 1 | 1 | 3 | 2 | 2 | 2 | 4 | Miscalculated chair cost, subtotal, tax, final total, and budget status; it also counted tax twice and reached the output limit. |\n",
        "| Llama 3.2 3B | P7 | 4 | 4 | 5 | 4 | 4 | 5 | 5 | Professional and within the word limit, but it did not directly address the damaged cartons and made a broad assurance about preventing recurrence. |\n",
        "| Qwen3 4B Instruct | P1 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Produced a compact 36-word summary containing every required fact. |\n",
        "| Qwen3 4B Instruct | P2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Correctly classified all five inquiries in the requested two-column table. |\n",
        "| Qwen3 4B Instruct | P3 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Returned valid, readable JSON with the exact schema and correct values. |\n",
        "| Qwen3 4B Instruct | P4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Used the furniture terms correctly, stayed within 120–160 words, included the measurement caveat, and avoided a code-compliance claim. |\n",
        "| Qwen3 4B Instruct | P5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Clearly separated known and unknown information and refused to invent warranty, certification, or capacity claims. |\n",
        "| Qwen3 4B Instruct | P6 | 5 | 4 | 4 | 3 | 5 | 5 | 4 | Correctly calculated the $21,330 subtotal, $1,493.10 tax, and $23,773.10 final total, but the 350-token limit cut off the remaining-budget statement. |\n",
        "| Qwen3 4B Instruct | P7 | 3 | 4 | 4 | 5 | 2 | 5 | 5 | Clear, empathetic, and within the word limit, but invented a logistical reason for the split shipment and promised an update by an unconfirmed deadline. |\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Average Quality Scores by Model\n",
        "\n",
        "| Model | Accuracy | Helpfulness | Format | Completeness | Grounding | Clarity | Speed | Overall average |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        "| Llama 3.2 3B | 3.43 | 3.71 | 4.43 | 4.00 | 3.86 | 4.00 | 4.86 | **4.04** |\n",
        "| Qwen3 4B Instruct | 4.71 | 4.71 | 4.71 | 4.71 | 4.57 | 5.00 | 4.86 | **4.76** |\n",
        "\n",
        "Qwen earned the higher overall quality score because it handled domain terminology, unsupported product claims, and arithmetic more reliably. Llama performed very well on the simpler summarization, classification, JSON, and tone tasks, but its failures on P5 and P6 were important because those mistakes could affect product advice or pricing. The evaluation also showed that neither model should be trusted without review: Qwen’s P7 response added an unsupported reason and deadline, while Llama’s P5 response invented a certification."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Performance Summary\n",
        "\n",
        "| Model | Average response time | Average tokens/sec | Average output tokens | Maximum peak GPU memory | Practical observation |\n",
        "|---|---:|---:|---:|---:|---|\n",
        "| Llama 3.2 3B | 5.57 s | 152.86 | 138.43 | 3,375 MiB | Fastest model and lowest measured GPU use. |\n",
        "| Qwen3 4B Instruct | 5.77 s | 118.34 | 137.14 | 3,971 MiB | Slightly slower and larger, but still responsive on this laptop. |\n",
        "\n",
        "The 0.20-second difference in average response time was small in normal use. Llama generated approximately 29% more tokens per second than Qwen and used 596 MiB less peak GPU memory. Even so, both models used far less than the available 16,303 MiB of VRAM, so hardware capacity was not a practical limitation in this test."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Part 7: Trade-Off and Recommendation\n",
        "\n",
        "The main trade-off in this experiment was speed and memory efficiency versus output reliability. Llama 3.2 3B was the faster and lighter option. It averaged 5.57 seconds per response and 152.86 tokens per second, compared with 5.77 seconds and 118.34 tokens per second for Qwen3 4B Instruct. Llama also peaked at 3,375 MiB of GPU memory, while Qwen reached 3,971 MiB. On this laptop, however, the extra 596 MiB used by Qwen was not a meaningful limitation because the GPU provides 16,303 MiB of VRAM. Both models were fast enough for an interactive office assistant.\n",
        "\n",
        "Output quality changed the recommendation. Qwen received an overall quality score of 4.76, compared with 4.04 for Llama. It correctly handled the furniture calculation, used the domain terminology more accurately, and avoided inventing specifications in the hallucination-sensitive prompt. Llama’s false certification claim and multiple arithmetic errors would create unacceptable risks if the output were sent directly to a client. Qwen was not perfect: its customer-service response invented a logistical explanation and promised a next-day update that had not been confirmed. This means human review remains necessary even for the stronger model.\n",
        "\n",
        "I would select Qwen3 4B Instruct for this local use case. Its modest decrease in speed is acceptable in exchange for the stronger accuracy, grounding, and instruction following observed across the seven prompts. Llama would still be useful when maximum speed matters and the task is simple, such as producing an internal first draft or basic classification. The downloaded packages are compact, quantized builds; Qwen was identified as Q4_K_M, while the exact quantization of the default Llama package was not separately verified. Because this was a multi-model comparison rather than a quantization sweep, the experiment cannot isolate how much quantization changed quality or speed.\n",
        "\n",
        "A hosted API could still be the better choice for complex reasoning, broader knowledge, enterprise support, or work where a mistake could have a serious financial or client impact. Local deployment is practical for routine, low-risk assistance because it supports privacy, offline access, and predictable cost, but it does not remove the need for product-document verification or professional judgment."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Reflection on the Results\n",
        "\n",
        "Llama was measurably faster, although the average response-time difference was only 0.20 seconds and was unlikely to feel important during ordinary use. Qwen followed the formatting and word-count instructions more consistently overall. Llama invented a GREENGUARD Gold certification in P5, while Qwen properly stated that none of the requested product claims could be verified. Qwen also completed the furniture subtotal, tax, and final-total calculations correctly; Llama made several arithmetic errors and incorrectly described the order as over budget. GPU memory use was practical for both models, since neither approached the available 16,303 MiB of VRAM. Based on this balance, Qwen is the better local option for the selected use case, while a hosted API would remain preferable for more demanding or high-risk tasks."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Conclusion\n",
        "\n",
        "Running both models locally made the trade-off much clearer than comparing model descriptions alone. Llama 3.2 3B was faster and more memory-efficient, and it handled several straightforward tasks very well. At the same time, its mistakes on the product-specification and cost-calculation prompts showed why speed cannot be the only deciding factor. Qwen3 4B Instruct required slightly more memory and generated text more slowly, but it produced the more dependable results and earned the stronger quality score.\n",
        "\n",
        "For this interior design and furniture sales use case, I would choose Qwen as the local model for summaries, classifications, structured project notes, and early drafts. The tested computer ran it comfortably, so the additional memory use did not create a real disadvantage. I would still review every response before using it with a client, especially when the task involves product specifications, prices, deadlines, or commitments. Overall, the project showed that a small local model can be practical and useful, but the right choice depends on the type of error the organization can accept, not only on which model responds fastest."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "458d4a79",
      "metadata": {},
      "source": [
        "## AI Usage Disclosure\n",
        "\n",
        "AI was used as a support tool for organizing the notebook structure, drafting the fictional prompt set, and generating code for local benchmarking. I installed the runtime, downloaded and ran the models on my own computer, reviewed the outputs, verified the benchmark results, assigned the quality scores, and made the final model recommendation based on my interpretation of the experiment."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.14.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
