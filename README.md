[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🔥 Ember</h1>
  <p align="center">
    Implement the internals of modern AI systems from scratch — Transformers, vLLM, TRL, and beyond.
  </p>
  <p align="center">
    <em>Read the paper, then write the code. No GPU required.</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#BGWH123/Ember&Date">
      <img src="https://img.shields.io/github/stars/BGWH123/Ember?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 🧠 What is Ember?

**101 coding problems** + **13 chapters of interview Q&A (八股文)** + **mock interview mode**.

Write the implementation, a local grading service runs the tests, you see what broke. Read the 八股文 to nail the theory questions. Use mock interview to mix coding + theory on the fly. That's it.

The 101 problems cover what's actually inside Transformers, vLLM, TRL, diffusion models, and GNNs — attention variants, training tricks, inference kernels, alignment algorithms, graph neural networks. No GPU needed.

### Who is this for?

- **Preparing for ML interviews** — practice implementing core components under test, not just reading about them
- **Learning by building** — if you learn best by writing code rather than watching lectures, this is your gym
- **Deepening your understanding** — you've used `nn.MultiheadAttention`, now write it yourself

### Features

- **Browser editor** — Monaco with Python syntax highlighting, no IDE setup
- **Instant feedback** — submit and see pass/fail per test case in seconds
- **Reference solutions** — compare after your own attempt, with line-by-line annotations
- **八股文 (Interview Q&A)** — 13 chapters covering LLM fundamentals, fine-tuning, RLHF, distillation, distributed training, and more
- **Mock interview** — random mix of theory questions and coding problems for realistic interview practice
- **Learning paths** — curated problem sequences from Transformer internals to GNNs
- **Progress tracking** — solved count and attempt history, persisted across sessions
- **AI Help** — optional AI-powered hints via any OpenAI-compatible API (configure in `.env` or per-user in the UI)
- **Fully local** — nothing leaves your machine (unless you opt into AI Help)

### Tech Stack

| Layer        | Technology                                                                           |
| ------------ | ------------------------------------------------------------------------------------ |
| Frontend     | Next.js + Monaco Editor + Tailwind CSS                                               |
| Backend      | FastAPI grading service                                                              |
| Judge Engine | [torch_judge](https://github.com/duoan/TorchCode) — executes and validates 101 problems with 3–5 test cases each |
| Storage      | SQLite (progress tracking)                                                           |

---

## 📢 News

- **[2026/05/01]** Expanded to **101 problems** across **22 categories** — from attention mechanisms to SSM, MoE, and graph neural networks. 🔥
- **[2026/04/28]** **八股文 (Interview Q&A)** — 13 chapters with 200+ theory questions covering LLM fundamentals, RLHF, distillation, and distributed training. 🔥
- **[2026/04/25]** **Mock interview mode** — random mix of theory questions and coding problems for realistic interview practice. 🔥
- **[2026/04/20]** New GNN learning path — 8 problems covering GCN, GAT, GIN, MPNN, GraphSAGE, link prediction, and graph autoencoders. 🔥
- **[2026/04/20]** New UI redesign with OKLch color system, dark mode, and Geist typography. 🔥
- **[2026/04/13]** Submission history — review all your past attempts per problem.
- **[2026/04/10]** AI Help — optional AI-powered hints via any OpenAI-compatible API. 🔥
- **[2026/04/09]** Initial release 🎉

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:

| Dependency | Minimum Version | Verify Command |
|------------|-----------------|----------------|
| Python     | 3.11            | `python --version` or `python3 --version` |
| Node.js    | 18              | `node --version` |
| npm        | (bundled with Node.js) | `npm --version` |
| Git        | any             | `git --version` |

**Platform Notes**
- **macOS**: Python may not be pre-installed. We recommend [Homebrew](https://brew.sh/) (`brew install python node`) or [uv](https://docs.astral.sh/uv/).
- **Windows**: Use PowerShell (recommended) or CMD. Ensure Python and Node.js are in your system `PATH`.
- **Linux**: Most distributions ship Python 3.11+. If not, use your package manager or [pyenv](https://github.com/pyenv/pyenv).

---

### Installation

We provide four ways to get Ember running. Pick the one that fits your workflow.

#### Option A — One-liner Script (Recommended)

This is the fastest way. The script automatically creates a Python virtual environment (prefers `uv`, falls back to `python -m venv`), installs all Python and Node.js dependencies, and prints the start command.

**macOS / Linux**

```bash
# 1. Clone the repository
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. Run the setup script
./setup.sh

# 3. Start the dev server
npm run dev
```

**Windows (PowerShell)** — run as Administrator if you encounter permission issues:

```powershell
# 1. Clone the repository
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. Run the setup script
.\setup.ps1

# 3. Start the dev server
npm run dev
```

**Windows (CMD)**:

```cmd
git clone https://github.com/BGWH123/Ember.git
cd Ember
setup.bat
npm run dev
```

<details>
<summary>What does the setup script do? (click to expand)</summary>

1. Checks for `uv` (fast Python package manager). If found, it uses `uv` to create `.venv` and install dependencies. If not, it falls back to `python -m venv`.
2. Installs the project in editable mode (`pip install -e ".[dev]"`).
3. Runs `npm install` to install frontend dependencies.
4. Prints a success message with the next steps.

</details>

> **Note**: When `.venv` exists, `npm run dev` automatically prefers that project-local Python. If `.venv` is missing, it falls back to the current shell's `python`.

---

#### Option B — Conda

If you already use Conda for environment management:

```bash
# 1. Clone
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. Create and activate environment
conda create -n pyre python=3.11 -y
conda activate pyre

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Install Node.js dependencies
npm install

# 5. Start
npm run dev
```

> Remember to `conda activate pyre` every time before running `npm run dev`.

---

#### Option C — Manual (venv)

For full control over the environment:

**macOS / Linux**

```bash
# 1. Clone
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. Create virtual environment
# Using uv (recommended):
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using standard venv:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -e ".[dev]"

# 3. Install frontend dependencies
npm install

# 4. Start
npm run dev
```

**Windows (PowerShell / CMD)**

```powershell
git clone https://github.com/BGWH123/Ember.git
cd Ember

python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

npm install
npm run dev
```

---

#### Option D — Docker

If you prefer containerized deployment or want to avoid installing Python/Node.js locally:

```bash
# 1. Clone
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. Build and run
docker compose up --build
```

Then open `http://localhost:3000`.

- **Grading service** runs inside the `grading` container and exposes port `8000`.
- **Web app** runs inside the `web` container and exposes port `3000`.
- Progress is persisted in a Docker volume (`pyre_data`).

To stop:
```bash
docker compose down
```

To reset all data (including progress):
```bash
docker compose down -v
```

> **Troubleshooting Docker**:
> - If you get port conflicts, check that nothing else is using ports `3000` and `8000`.
> - On Windows, ensure Docker Desktop is running before executing `docker compose up`.

---

### Verify Installation

Once running, you should see two services:

| Service          | URL                     | What to check                          |
|------------------|-------------------------|----------------------------------------|
| Grading service  | `http://localhost:8000` | Open in browser → should show FastAPI docs |
| Web app          | `http://localhost:3000` | Should load the problem list           |

If both are accessible, you're all set. Pick a problem and start coding!

---

### AI Help (Optional)

To enable server-side AI hints, copy the example environment file and fill in your API details:

```bash
cp web/.env.example web/.env
```

Edit `web/.env`:

```bash
AI_HELP_BASE_URL=https://api.openai.com/v1
AI_HELP_API_KEY=sk-...
AI_HELP_MODEL=gpt-4o-mini
```

Any OpenAI-compatible endpoint works (OpenAI, Anthropic via proxy, Ollama, etc.). Users can also configure their own API key in the UI if no server-side config is set.

---

### Troubleshooting / FAQ

<details>
<summary><strong>Q: Setup script fails with "python not found"</strong></summary>

Make sure Python 3.11+ is installed and on your `PATH`. On Windows, you may need to use `py` instead of `python`. You can also manually create a `.venv` (see Option C).

</details>

<details>
<summary><strong>Q: npm install fails / hangs</strong></summary>

- Check your Node.js version: `node --version` (must be >= 18).
- Try clearing the npm cache: `npm cache clean --force`.
- If you're behind a proxy, configure npm: `npm config set proxy http://...`.

</details>

<details>
<summary><strong>Q: Backend starts but frontend shows "Failed to connect"</strong></summary>

Check that the grading service is actually running on port `8000`:
```bash
curl http://localhost:8000
```
If it's not running, check the logs in `logs/backend.log` or run the backend manually:
```bash
python -m grading_service.main
```

</details>

<details>
<summary><strong>Q: Docker build fails with "public directory not found"</strong></summary>

Make sure your `web/` directory has a `public/` folder. If not, create an empty one:
```bash
mkdir -p web/public
```
This is sometimes needed for Next.js static exports.

</details>

<details>
<summary><strong>Q: Can I run without Docker?</strong></summary>

Yes — Options A, B, and C all run natively without Docker.

</details>

<details>
<summary><strong>Q: Do I need a GPU?</strong></summary>

No. All problems run on CPU. A GPU is only needed if you want to train large models outside of Ember.

</details>

## 📋 Problem Set

**101 problems** across **22 categories** — every problem has **3–5 test cases** validated by a local FastAPI grading service.

| Category | Count | Problems |
|---|---|---|
| **Attention Mechanisms** | 14 | Scaled Dot-Product, Multi-Head, Causal, Cross, GQA, Sliding Window, Linear, Flash, Differential, MLA, Attention Mask, CLIP, Einsum, Cosine Similarity |
| **Fundamental Components** | 14 | ReLU, GELU, SwiGLU, Leaky ReLU, Softmax, Dropout, Embedding, Linear, Kaiming Init, Linear Regression, MLP, Residual Connection, LayerScale, Broadcasting |
| **Loss Functions** | 11 | Cross Entropy, Label Smoothing, Focal Loss, Contrastive Loss, Binary Cross Entropy, MSE, KL Divergence, L1/L2 Regularization, DPO, GRPO, PPO |
| **Graph Neural Networks** | 7 | GCN, Graph Readout, GAT, GIN, MPNN, GraphSAGE, Link Prediction, Graph Autoencoder |
| **Activation Functions** | 7 | ReLU, GELU, SwiGLU, Leaky ReLU, Sigmoid (BCE), Softmax |
| **Transformer Components** | 6 | GPT-2 Block, ViT Patch, ViT Block, Transformer Encoder, Transformer Decoder, CLIP Model |
| **Optimizers & LR** | 5 | Adam, Cosine LR, Gradient Clipping, L1/L2 Regularization |
| **Sampling & Decoding** | 5 | Top-k, Beam Search, Speculative Decoding, BPE, Multi-Token Prediction |
| **Regularization** | 5 | Dropout, L1, L2, Label Smoothing, LayerScale |
| **Efficient Training** | 4 | Mixed Precision, Gradient Accumulation, Activation Checkpointing, FSDP |
| **Normalization** | 4 | LayerNorm, BatchNorm, RMSNorm, adaLN-Zero |
| **Position Encoding & Embedding** | 3 | Sinusoidal PE, RoPE, ALiBi, NTK-aware RoPE |
| **Mixture of Experts** | 3 | MoE, MoE Load Balance, Multi-Token Prediction |
| **Diffusion & Flow** | 2 | Noise Schedule, DDIM Step, Flow Matching |
| **Quantization** | 2 | INT8 Quantization, QLoRA |
| **Inference Optimization** | 2 | KV Cache, Speculative Decoding |
| **Training Tricks** | 2 | Gradient Accumulation, Activation Checkpointing |
| **Tokenization** | 1 | BPE |
| **Multimodal** | 1 | CLIP Model |
| **Parameter-Efficient FT** | 1 | LoRA |
| **State Space Models** | 1 | Mamba SSM |
| **Reinforcement Learning** | 1 | PPO Loss |

### 📝 八股文 & Mock Interview

Beyond coding, Ember includes a full **interview Q&A (八股文)** module and **mock interview** mode:

**八股文 — 13 chapters, 200+ questions:**
| Chapter | Topics |
|---|---|
| LLM Fundamentals | Architecture (Encoder-Decoder / Causal / Prefix), training objectives, emergent capabilities |
| Advanced LLM | Repetition problem, LLaMA series, long-context techniques |
| Evaluation | POPE, MME, CHAIR, AMBER, benchmark design |
| Inference | KV cache, quantization, decoding strategies, vLLM |
| Fine-tuning | LoRA, QLoRA, full fine-tuning, instruction tuning |
| Training Data | Data collection, cleaning, deduplication, mixture of domains |
| Agent | Tool use, ReAct, planning, multi-agent systems |
| RLHF / PPO | Reward modeling, PPO, DPO, GRPO, preference optimization |
| LangChain | Chains, agents, memory, retrieval-augmented generation |
| Continual Pre-training | Domain adaptation, catastrophic forgetting |
| Distillation | Knowledge distillation, self-distillation |
| Distributed Training | Data parallel, tensor parallel, pipeline parallel, FSDP |
| GPU Memory | Gradient checkpointing, activation recomputation, offloading |

**Mock Interview:**
- Randomly draws theory questions from 八股文 + coding problems from the problem set
- Simulates realistic interview pacing (theory → coding → follow-up)
- Tracks your performance across sessions

### Learning Paths

Pick one based on what you're working toward:

| Path | Problems | Description |
|---|---|---|
| **Transformer Internals** | 12 | Activations → Normalization → Attention → GPT-2 Block |
| **Attention & Position Encoding** | 13 | Every attention variant + RoPE, ALiBi, NTK-RoPE |
| **Train a GPT from Scratch** | 15 | Embeddings → architecture → loss → optimizer → training tricks |
| **Inference & Distributed Training** | 9 | KV cache, quantization, sampling, tensor parallel, FSDP |
| **Alignment & Agent Reasoning** | 6 | Reward model → DPO → GRPO → PPO → MCTS |
| **Vision Transformer Pipeline** | 7 | Conv → patch embedding → ViT block |
| **Diffusion Models & DiT** | 5 | Noise schedule → DDIM → flow matching → adaLN-Zero |
| **LLM Frontier Architectures** | 7 | GQA, Differential Attention, MLA, MoE, Multi-Token Prediction |
| **Graph Neural Networks** | 8 | GCN → GAT → GIN → MPNN → GraphSAGE → Link Prediction → GAE |

```
Not sure where to start?

Fundamentals ──→ Transformer Internals ──→ Train a GPT from Scratch
                       │                          │
                       ▼                          ▼
              Attention & Position       Inference & Distributed
                       │                          │
                       ▼                          ▼
              LLM Frontier Archs         Alignment & Reasoning
                       │
               ┌───────┼───────┐
               ▼       ▼       ▼
     Vision Trans.  Diffusion  Graph Neural Networks
```

---

## ⚙️ Configuration

| Variable                | Default                   | Description                           |
| ----------------------- | ------------------------- | ------------------------------------- |
| `GRADING_SERVICE_URL` | `http://localhost:8000` | Grading service URL                   |
| `DB_PATH`             | `./data/ember.db`   | SQLite database for progress tracking |

Set in `web/.env.local` to override.

---

## 📁 Project Structure

```
ember/
├── web/                  # Next.js frontend
│   ├── src/app/          # Pages (problems, bagu, interview, paths)
│   ├── src/components/   # UI components
│   └── src/lib/          # Utilities, problem & path data
├── grading_service/      # FastAPI backend — runs tests per problem
├── torch_judge/          # Judge engine (101 problem definitions + test runner)
├── scripts/              # Build & test scripts (export problems, test grading)
└── package.json          # Dev scripts (runs frontend + backend concurrently)
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- **Submit a new problem** — open a PR with the problem definition and test cases in `torch_judge/`
- **Report a bug** — [open an issue](https://github.com/BGWH123/Ember/issues) with steps to reproduce
- **Fix a bug** — fork, fix, and submit a PR
- **Improve docs** — typos, clarifications, translations

Please open an issue first for larger changes so we can discuss the approach.

---

## ⭐ Star History

![GitHub stars](https://img.shields.io/github/stars/BGWH123/Ember?style=social)

[![Star History Chart](https://api.star-history.com/svg?repos=BGWH123/Ember&type=Date)](https://star-history.com/#BGWH123/Ember&Date)

---

## 🙏 Acknowledgements

Problem set and judge engine based on [TorchCode](https://github.com/duoan/TorchCode) by [duoan](https://github.com/duoan), licensed under MIT.

UI, learning paths, and additional features inspired by [pyre-code](https://github.com/whwangovo/pyre-code) by [whwangovo](https://github.com/whwangovo). Many thanks for the excellent reference implementation.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
