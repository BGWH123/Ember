<div align="center">

<img src="./apps/desktop/src/app/icon.svg" width="84" height="84" alt="Ember Logo" />

# Ember

**Build the Internals of Modern AI Systems from Scratch — One PyTorch Function at a Time.**

English | [简体中文](./README_CN.md)

[![Problems](https://img.shields.io/badge/PyTorch_Challenges-181-f97316?style=flat-square)](#what-you-will-implement-181-challenges)
[![Learning Paths](https://img.shields.io/badge/Learning_Paths-12-2563eb?style=flat-square)](#12-structured-learning-paths)
[![Theory Handbook](https://img.shields.io/badge/Theory_Tutorials-35_Bilingual-10b981?style=flat-square)](#all-in-one-ai-engineering-workspace)
[![Interview Bank](https://img.shields.io/badge/Interview_Q%26As-1%2C316-8b5cf6?style=flat-square)](#all-in-one-ai-engineering-workspace)
[![Architecture](https://img.shields.io/badge/Stack-Tauri_2_%C2%B7_Rust_%C2%B7_React_%C2%B7_PyTorch-0f172a?style=flat-square)](#system-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)

</div>

---

## Why Ember?

Reading papers and memorizing formulas is not enough when you are facing a blank editor in an **LLM, Multimodal, Diffusion, or Embodied AI interview**—or debugging a custom training loop at 2 AM. True mastery comes from implementing the mathematical kernels yourself and watching gradients flow through them.

**Ember** is a **local-first, offline-ready desktop training ground** for AI engineers and researchers. It combines a LeetCode-style **PyTorch coding arena** with a comprehensive **AI theory handbook**, **curated learning paths**, and a **mock interview simulator** inside a fast native desktop app built on **Tauri 2** and **Rust**.

- **Real PyTorch Execution, Zero Mocks**: Every submission runs against a real, bounded PyTorch subprocess judge that verifies tensor shapes, exact numerical parity against reference formulas, edge cases, and `autograd` gradient routing (`detach` boundaries, stop-gradients, and Straight-Through Estimators).
- **Instant Offline Browsing**: Browse all **181 coding challenges**, reference implementations, mathematical explanations, **12 learning paths**, **35 bilingual tutorials**, and **1,316 interview questions** out-of-the-box—no Python installation or internet connection required.
- **Pluggable Local Evaluator**: Attach your local Python environment (or a cryptographically signed standalone evaluator module) whenever you are ready to run and benchmark code locally on your machine.
- **Multi-Tab Workspace & AI Tutor**: Work across multiple problems simultaneously with persistent drafts, inspect per-test execution latency, or connect your own LLM endpoint for Socratic debugging hints without spoiling the solution.

---

## All-in-One AI Engineering Workspace

| Module | What's Inside | Highlights |
| :--- | :--- | :--- |
| **Coding Arena** (`/problems`) | **181 executable PyTorch tasks** (`32 Easy` · `84 Medium` · `65 Hard`) across 19 categories | Bilingual problem statements, collapsible mathematical theory, Mermaid architecture diagrams, hints, runnable demos, and 4+ rigorous unit tests per task |
| **Learning Paths** (`/paths`) | **12 guided sequences** from foundational layers to frontier architectures | Step-by-step progression with prerequisite tracking and completion badges |
| **Knowledge Map** (`/curriculum`) | **8-stage AI Engineer Curriculum** covering 32 core competency modules | Bridges hands-on PyTorch exercises, Hot 50 interview breakdowns, and deep-dive handbook chapters in one unified graph |
| **Theory Handbook** (`/bagu`) | **35 bilingual (ZH/EN) tutorials** + **3 deep-dive technical blogs** + **9 Embodied AI interview volumes** | Offline MathJax rendering, syntax-highlighted companion scripts, and full-text search |
| **Hot 50** (`/hot`) | **50 highest-frequency AI interview questions** across 9 domains | Structured answers with key takeaways, common pitfalls, and primary paper citations |
| **Mock Interview** (`/interview`) | **Dual-mode simulator**: 1,316 theory flashcards + 181 live coding draws | Think-then-reveal mode for conceptual rounds; one-click jump to the live editor for coding rounds |

---

## What You Will Implement (181 Challenges)

Ember's 181 coding problems cover the full stack of modern AI systems—from foundational tensor calculus to the exact equations powering **DeepSeek-V3/R1, LLaMA-3, Qwen2-VL, Flux/SD3, Sora, DreamerV3, and OpenVLA**.

<details open>
<summary><b>1. Attention Mechanisms & Long-Context Modeling (24 Problems)</b></summary>

- **Core Attention**: Scaled Dot-Product Attention (`attention`), Multi-Head Attention (`mha`), Causal Self-Attention (`causal_attention`), Multi-Head Cross-Attention (`cross_attention`), Padding Masks (`attention_mask`), QK-Norm Attention (`qk_norm_attention`).
- **KV-Efficient & Frontier Attention**: Grouped-Query Attention (`gqa`), DeepSeek Multi-Head Latent Attention (`mla`), **MLA Decoupled RoPE & Weight Matrix Absorption** (`mla_rope_absorption`), Differential Attention (`diff_attention`), Attention Logit Soft-Capping & StreamingLLM Sink Mask (`attn_softcap_and_sink`).
- **IO-Aware, Distributed & Sparse Attention**: Tiled FlashAttention (`flash_attention`), Distributed Ring Attention (`ring_attention`), vLLM PagedAttention (`paged_attention`), Sliding Window Attention (`sliding_window`), Block Top-K Sparse Attention (`block_topk_attention`), 2D Axial Self-Attention (`axial_attention`).
- **Linear & Recurrent Attention**: Kernel Linear Attention (`linear_attention`), **Chunkwise Parallel Causal Linear Attention** (`chunkwise_linear_attention`), **DeltaNet Erase-and-Write Recurrence** (`deltanet_recurrence`).
- **Positional Encodings**: Sinusoidal PE (`sinusoidal_pe`), Rotary Position Embedding (`rope`), NTK-Aware RoPE (`ntk_rope`), **YaRN Frequency Ramp & Temperature Scaling** (`yarn_rope`), ALiBi Linear Biases (`alibi`).

</details>

<details open>
<summary><b>2. Mixture-of-Experts (MoE), SSMs & Frontier LLM Blocks (14 Problems)</b></summary>

- **Mixture-of-Experts**: Sparse Top-K MoE Layer (`moe`), **DeepSeek-V3 Aux-Loss-Free Sigmoid Router with Dynamic Bias** (`deepseek_moe_router`), Noisy Top-K Gating (`noisy_topk_gating`), Switch Load-Balancing Loss (`moe_load_balance`), Router Z-Loss (`moe_router_z_loss`), **Mixture-of-Depths (MoD) & Expert-Choice Routing** (`mod_and_expert_choice`).
- **Transformer & State-Space Backbones**: GPT-2 Pre-Norm Block (`gpt2_block`), Complete Transformer Encoder (`transformer_encoder`) & Decoder (`transformer_decoder`), SwiGLU MLP (`mlp`, `swiglu`), GeGLU FFN (`geglu`), Mamba Selective State-Space Model Step (`mamba_ssm`).
- **Multi-Token Prediction**: Meta Parallel MTP Loss (`multi_token_prediction`), **DeepSeek-V3 Sequential Causal MTP Module** (`deepseek_mtp_module`).

</details>

<details open>
<summary><b>3. Post-Training, RLHF, Alignment & Reasoning Models (18 Problems)</b></summary>

- **Preference Optimization**: Direct Preference Optimization (`dpo_loss`), **SimPO Reference-Free Length-Normalized Loss** (`simpo_loss`), **ORPO Odds-Ratio Preference Loss** (`kto_orpo_loss`), **IPO Squared-Margin & KTO Prospect-Theory Losses** (`ipo_and_kto_loss`), Bradley-Terry Reward Model (`reward_model`).
- **RL for LLMs & Reasoning**: PPO Clipped Surrogate (`ppo_loss`), DeepSeek **GRPO Group-Relative Loss** (`grpo_loss`), **DAPO Clip-Higher & Token-Level Policy Optimization** (`dapo_loss`), **RLOO Leave-One-Out Advantage & PRM Step Aggregation** (`rloo_and_prm_aggregation`), **Schulman KL Estimators ($k_1, k_2, k_3$)** (`kl_estimators_rlhf`), **On-Policy Distillation Generalized JSD (GKD)** (`gkd_generalized_jsd`).
- **Tree Search & Deep RL**: AlphaZero/Reasoning **PUCT Selection** (`puct_selection`), UCB1 MCTS (`mcts_search`), **Generalized Advantage Estimation (GAE)** (`generalized_advantage_estimation`), **SAC Squashed Gaussian Policy & Tanh Jacobian Correction** (`sac_squashed_gaussian`), **Offline RL: CQL & IQL Expectile Loss** (`cql_iql_loss`), **Double DQN & TD3 Clipped Double-Q Targets** (`double_dqn_td3_target`).

</details>

<details open>
<summary><b>4. Diffusion, Flow Matching, Discrete Latents & Video/Image Generation (18 Problems)</b></summary>

- **Flow Matching & Diffusion Foundations**: Rectified Flow Matching Loss (`flow_matching`), **Heun 2nd-Order ODE Sampler** (`flow_heun_sampler`), **DDPM Posterior & $v$-Prediction Conversion** (`ddpm_posterior_and_vpred`), DDIM Deterministic Step (`ddim_step`), Noise Schedules (`noise_schedule`), **Min-SNR-$\gamma$ Weighting & Karras EDM Preconditioning** (`min_snr_and_edm_precond`), **Classifier-Free Guidance (CFG) & Std Rescaling** (`classifier_free_guidance`).
- **Generative Post-Training & Distillation**: **Flow-GRPO Reverse SDE & Velocity-Space KL** (`flow_grpo_sde_loss`), **Diffusion-DPO & REPA Representation Alignment** (`diffusion_dpo_repa_loss`), **Consistency Distillation & Pseudo-Huber Loss** (`consistency_distillation_loss`), **Masked Diffusion LM (MDLM) & Block-Causal Mask** (`mdlm_and_block_diffusion`).
- **Visual Tokenizers & DiT / Video Architectures**: **MMDiT Joint Image-Text Attention** (`mmdit_joint_attention`), DiT `adaLN-Zero` Conditioning (`adaln_zero`), **Video VAE Temporal Causal Conv3D** (`causal_conv3d`), **VAE Reparameterization & $\beta$-ELBO** (`vae_elbo_reparam`), **VQ-VAE Vector Quantizer with STE** (`vqvae_ste`), **Finite Scalar Quantization (FSQ)** (`finite_scalar_quantization`).

</details>

<details open>
<summary><b>5. PEFT, Quantization, Inference Serving & Distributed Training (19 Problems)</b></summary>

- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA (`lora`), **DoRA Weight-Decomposed Low-Rank Adaptation** (`dora_linear`), **PiSSA Principal Singular Value Init** (`pissa_init`), QLoRA (`qlora`).
- **Model & KV Cache Quantization**: Symmetric INT8 Linear (`int8_quantization`), **Group-Wise Asymmetric INT4** (`groupwise_int4_quant`), **QLoRA NF4 Codebook & Double Quantization** (`nf4_double_quantization`), **SmoothQuant Activation-Weight Migration** (`smoothquant_migration`), **AWQ Salience Scaling & GPTQ Hessian Error Compensation** (`awq_and_gptq_quant`), **KIVI Asymmetric KV Cache Quantization** (`kivi_kv_cache_quant`).
- **Inference & Decoding**: KV Cache Attention (`kv_cache`), **H2O Heavy-Hitter & Sink KV Eviction** (`h2o_kv_eviction`), Speculative Decoding (`speculative_decoding`), **Tree Speculative Attention Mask (Medusa/EAGLE)** (`tree_speculative_mask`), Top-k/Top-p Sampling (`topk_sampling`), **Min-P & Repetition Penalty Filter** (`min_p_sampling`), Beam Search (`beam_search`), Temperature Softmax (`softmax_temperature`).
- **Distributed & Pretraining Systems**: Tensor Parallel MLP (`tensor_parallel`), FSDP Step (`fsdp_step`), **ZeRO-1/2/3 Memory Accounting & 1F1B Pipeline Bubble** (`zero_memory_accounting`), Mixed Precision Step (`mixed_precision`), Activation Checkpointing (`activation_checkpointing`), **Intra-Document Causal Packing Mask** (`document_packing_mask`), Sequence Packing (`sequence_pack`), **Chinchilla Compute-Optimal Scaling Law** (`chinchilla_scaling`).

</details>

<details open>
<summary><b>6. Multimodal, RAG, World Models, 3D Vision & Embodied Robotics (20 Problems)</b></summary>

- **Vision-Language & Multimodal**: CLIP Contrastive Model (`clip_model`), **SigLIP Pairwise Sigmoid Loss** (`siglip_loss`), **Qwen2-VL Multimodal 3D RoPE (M-RoPE)** (`multimodal_3d_rope`), **Cosmos-3 Mixture-of-Transformers (MoT) Asymmetric Attention** (`mot_asymmetric_attention`), **Pixel Unshuffle Visual Token Compression** (`perceiver_resampler`), ViT Patch Embedding (`vit_patch`) & Block (`vit_block`).
- **RAG & Retrieval**: **BM25 Scoring & Reciprocal Rank Fusion (RRF)** (`bm25_rrf_retrieval`), **ColBERT Late-Interaction MaxSim** (`matryoshka_colbert_loss`), InfoNCE Contrastive Loss (`contrastive_loss`), Cosine Similarity (`cosine_similarity`).
- **World Models**: **DreamerV3 Symlog Transform & $\lambda$-Return** (`dreamer_world_model`), **JEPA Latent Prediction & VICReg Anti-Collapse Regularization** (`jepa_and_vicreg_loss`).
- **3D Vision & Embodied Robotics**: **NeRF Volume Rendering** (`nerf_volume_rendering`), **3D Gaussian Splatting Covariance & Quaternion Rotation** (`gaussian_splatting_cov3d`), **2D Box IoU & NMS** (`box_iou_nms`), **Kabsch 3D Point Cloud Rigid SVD Alignment** (`kabsch_pointcloud_align`), **Pinhole Depth Unprojection & Farthest Point Sampling (FPS)** (`pointcloud_fps_and_unproject`), **VLA Action Chunking (ACT) Temporal Ensemble & OpenVLA Tokenizer** (`act_openvla_actions`), **Damped Least Squares IK & Joint PD Control** (`damped_least_squares_ik`), **Linear Kalman Filter Predict & Update** (`kalman_lqr_control`), **SVD PCA & Cross-Entropy Method (CEM) Planner** (`kmeans_pca_and_cem`).

</details>

<details>
<summary><b>7. Optimizers, Normalization, Tokenization, Evaluation, GNNs & Core Layers (68 Problems)</b></summary>

- **Optimizers & LR Schedules**: Adam (`adam`), **AdamW Decoupled Weight Decay** (`adamw`), **Lion Sign-Momentum Optimizer** (`lion_optimizer`), **Muon Newton-Schulz Matrix Orthogonalization** (`muon_orthogonalize`), SGD with Momentum (`sgd_momentum`), Cosine Warmup LR (`cosine_lr`), **Warmup-Stable-Decay (WSD) LR Schedule** (`wsd_lr_schedule`), Gradient Norm Clipping (`gradient_clipping`), Gradient Accumulation (`gradient_accumulation`).
- **Normalization & Residual Scaling**: LayerNorm (`layernorm`), RMSNorm (`rmsnorm`), BatchNorm (`batchnorm`), **GroupNorm** (`groupnorm`), **Dynamic Tanh (DyT)** (`dynamic_tanh`), **DeepNorm Residual Scaling** (`deepnorm_residual`), LayerScale (`layerscale`), Stochastic Depth (`stochastic_depth`), Dropout (`dropout`), Kaiming Init (`weight_init`).
- **Tokenization & Benchmarking**: Byte-Pair Encoding (`bpe`), **WordPiece Greedy Tokenizer** (`wordpiece_tokenizer`), **Unigram Viterbi Segmentation** (`unigram_viterbi`), **Unbiased Pass@k Estimator** (`pass_at_k`), **Expected Calibration Error (ECE) & Brier Score** (`expected_calibration_error`).
- **Graph Neural Networks**: GCN (`gcn_layer`), GAT (`gat_layer`), GraphSAGE (`graphsage_layer`), GIN (`gin_layer`), MPNN (`mpnn_layer`), Graph Readout (`graph_readout`), Graph Autoencoder (`gae`), Link Prediction (`link_prediction`).
- **Activations, Losses & Neural Primitives**: ReLU (`relu`), Leaky ReLU (`leaky_relu`), GELU (`gelu`), Sigmoid (`sigmoid`), Tanh (`tanh`), Softmax (`softmax`), Stable LogSumExp & Log-Softmax (`logsumexp_logsoftmax`), Cross-Entropy (`cross_entropy`), BCE (`binary_cross_entropy`), MSE (`mse_loss`), Focal Loss (`focal_loss`), KL Divergence (`kl_divergence`), Label Smoothing (`label_smoothing`), L1/L2 Regularization (`l1_regularization`, `l2_regularization`), Conv2d (`conv2d`), Depthwise Separable Conv (`depthwise_conv`), MaxPool2d (`max_pool2d`), LSTM Cell (`lstm_cell`), GRU Cell (`gru_cell`), Linear (`linear`), Embedding (`embedding`), Linear Regression (`linear_regression`), Einsum (`einsum_ops`), Broadcasting (`broadcasting`), One-Hot (`one_hot_encoding`), Residual Connection (`residual_connection`).

</details>

---

## 12 Structured Learning Paths

Not sure where to start? Follow one of the 12 curated tracks in the **Paths** tab:

1. **Transformer Internals** — Activations, normalizations, attention, GeGLU/SwiGLU, and complete Transformer blocks.
2. **Attention & Position Encoding** — From scaled dot-product and RoPE to FlashAttention, YaRN, DeltaNet, and Block Top-K sparse attention.
3. **Train a GPT from Scratch** — Embeddings, document packing, AdamW/Lion/Muon, WSD schedule, and mixed-precision training.
4. **Inference & Distributed Training** — KV cache, KIVI, H2O eviction, tree speculative decoding, TP, FSDP, and ZeRO memory accounting.
5. **Alignment & Agent Reasoning** — Reward modeling, Schulman KL, PPO, DPO, SimPO, ORPO, IPO/KTO, GRPO, DAPO, RLOO/PRM, GAE, and PUCT.
6. **Vision Transformer Pipeline** — Convolutions, ViT patch embeddings, Axial Attention, Pixel Unshuffle, and SigLIP.
7. **Diffusion Models & DiT** — Noise schedules, DDPM/DDIM, Min-SNR/EDM, CFG, Flow Matching, Heun ODE, Causal Conv3D, MMDiT, and Flow-GRPO.
8. **LLM Frontier Architectures** — GQA, Differential Attention, MLA + Matrix Absorption, DeepSeek MoE, MoD, MDLM, and DeepSeek MTP.
9. **Graph Neural Networks** — GCN, GAT, GIN, MPNN, GraphSAGE, Graph Readout, Link Prediction, and GAE.
10. **PEFT, Quantization & Pretraining Scaling** — LoRA, DoRA, PiSSA, QLoRA, NF4, INT8/INT4, SmoothQuant, AWQ/GPTQ, Chinchilla, Pass@k, and ECE.
11. **RAG, World Models & Agentic AI** — BM25 + RRF, ColBERT MaxSim, SigLIP, M-RoPE, MoT, DreamerV3, and JEPA/VICReg.
12. **Embodied AI, 3D Vision & Control** — Box IoU/NMS, Depth Unprojection/FPS, Kabsch SVD, NeRF, 3DGS, DLS IK, Kalman Filter, ACT/OpenVLA, SAC, and CQL/IQL.

---

## Quick Start

### 1. Prerequisites

- **Node.js** 22+
- **Rust** stable toolchain (MSVC on Windows with Visual Studio C++ Build Tools)
- Platform prerequisites for [Tauri 2](docs/fern/pages/desktop/installation/install-desktop.mdx)

### 2. Launch the Desktop App (No Python Needed for Browsing)

```bash
npm ci
npm run dev
```

Noto Sans, Noto Sans SC, and Fira Code variable fonts are bundled offline inside the application.

### 3. Enable Live PyTorch Code Evaluation (Development Mode)

To execute and grade PyTorch submissions locally during development, create a virtual environment and point `EMBER_DEV_PYTHON` to it:

**Windows (PowerShell):**

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r modules/evaluator/requirements.txt
uv pip install --python .venv/Scripts/python.exe pytest
$env:EMBER_DEV_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
npm run dev
```

**macOS / Linux:**

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r modules/evaluator/requirements.txt
uv pip install --python .venv/bin/python pytest
EMBER_DEV_PYTHON="$(pwd)/.venv/bin/python" npm run dev
```

> **Note:** `EMBER_DEV_PYTHON` is accepted only in debug builds. Release installers verify Ed25519-signed standalone evaluator bundles. See the [Build from Source guide](docs/fern/pages/developer-guide/development/build-from-source.mdx).

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        Ember Desktop (Tauri 2)                          │
│   React 19 · TypeScript · Monaco Editor · MathJax 3 · Tailwind CSS      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ Tauri IPC (Zero HTTP Overhead)
┌───────────────────────────────────▼─────────────────────────────────────┐
│                        Rust Core (ember-core)                           │
│   • Catalog & Learning Paths Engine    • SQLite Progress & Draft Store  │
│   • Submission Idempotency & History   • Socratic AI Guidance Proxy     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ JSON-Lines Subprocess Protocol
┌───────────────────────────────────▼─────────────────────────────────────┐
│                Sandboxed Evaluator (ember-evaluator)                    │
│   • Ed25519 Manifest Verification      • Disposable Worker Processes    │
│   • CPU / Wall-Clock / RSS Memory Caps • Restricted Builtins & Imports  │
└─────────────────────────────────────────────────────────────────────────┘
```

| Directory | Responsibility |
| :--- | :--- |
| `apps/desktop` | React 19 UI, Monaco code workspace, Theory Handbook reader, and Tauri 2 shell |
| `crates/ember-core` | Problem catalog, learning paths, SQLite persistence, and AI tutor proxy |
| `crates/ember-evaluator` | Local subprocess judge protocol, Ed25519 module signature verification, and installer |
| `modules/evaluator` | Standalone Python/PyTorch test runner with per-task process isolation and resource budgets |
| `torch_judge/tasks` | **181 self-contained PyTorch task definitions** (descriptions, hints, theory, tests, and reference solutions) |
| `content/catalog` | Generated frontend/Rust catalogs (`problems.json`, `starters.json`, `solutions.json`, `paths.json`) |
| `scripts` | Catalog exporters, deduplication auditor (`audit_deduplication.py`), and desktop packaging scripts |

---

## Testing, Catalog Sync & Packaging

Refresh the catalogs after adding or editing tasks in `torch_judge/tasks/`:

```bash
npm run build:problems
npm run build:solutions
node scripts/run-python.js scripts/build_starters.py
node scripts/run-python.js scripts/audit_deduplication.py
```

Run the full verification suite (Rust core tests, Python sandboxed judge & mutation coverage tests, UI build, and Playwright end-to-end tests):

```bash
npm run check:core
npm run test:python
npm run build:ui
cargo build --release -p ember-core --features developer-tools
npm --prefix apps/desktop exec -- playwright install chromium
npm --prefix apps/desktop test
```

Build native desktop installers:

```bash
npm run build:windows   # Outputs .msi and .exe installers to target/release/bundle
npm run build:macos     # Outputs .app and .dmg bundles on macOS
```

---

## Documentation & Credits

- **Developer & Architecture Docs**: See [`docs/README.md`](docs/README.md) (`npm run docs:dev` to preview locally).
- **Acknowledgements**:
  - Problem runner inspired by and expanded from [TorchCode](https://github.com/duoan/TorchCode) (MIT License).
  - Offline theory tutorials and technical blogs vendored from [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer) by Ruofeng Yang (MIT License).
  - Embodied AI interview question bank vendored from [embodied-interview-qa](https://github.com/WinstonJQ/embodied-interview-qa) by WinstonJQ (MIT License).
- **License**: [MIT](./LICENSE).
