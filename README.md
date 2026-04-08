<div align="center">

```
     ██████╗ ██╗  ██╗ ██████╗ ███╗   ██╗
    ██╔══██╗╚██╗██╔╝██╔═══██╗████╗  ██║
    ███████║ ╚███╔╝ ██║   ██║██╔██╗ ██║
    ██╔══██║ ██╔██╗ ██║   ██║██║╚██╗██║
    ██║  ██║██╔╝ ██╗╚██████╔╝██║ ╚████║
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

### The ML Programming Language

**Write 50 lines of Axon. Get 500 lines of production Python.**

A declarative language that transpiles to PyTorch, TensorFlow, or JAX — replacing
thousands of lines of ML boilerplate with clean, readable syntax.

[![Version](https://img.shields.io/badge/version-0.1.2-blue?style=for-the-badge)](https://github.com/desenyon/axon-lang)
[![Tests](https://img.shields.io/badge/tests-225%20passed-brightgreen?style=for-the-badge)](tests/)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-purple?style=for-the-badge)](LICENSE)

<br>

**`16,000 lines`** of implementation &nbsp;·&nbsp; **`35`** block types &nbsp;·&nbsp; **`3`** backends &nbsp;·&nbsp; **`18`** examples &nbsp;·&nbsp; **`225`** tests

<br>

[Quick Start](#-quick-start) · [Examples](#-see-it-in-action) · [All 35 Blocks](#-all-35-block-types) · [Architecture](#-architecture) · [Install](#-install)

</div>

<br>

---

<br>

## ⚡ Quick Start

```bash
# Install
pip install -e .

# Create a new project
axon init my_project

# Compile Axon → Python
axon compile model.axon                    # Default: PyTorch
axon compile model.axon --backend jax      # Or target JAX
axon compile model.axon --backend tensorflow

# Validate syntax
axon check model.axon
axon check model.axon --semantic         # Deep ML pattern analysis

# Developer tools
axon fmt model.axon                       # Auto-format
axon lint model.axon                      # Static analysis (13 rules)
axon watch ./src                          # Auto-recompile on changes

# Interactive REPL
axon repl

# Start LSP server (for VS Code / editors)
axon lsp
```

<br>

## 🔥 See It in Action

<table>
<tr>
<td width="50%" valign="top">

### Axon — 12 lines

```axon
model ImageClassifier:
    backbone: resnet50(pretrained=true)
    pool: AdaptiveAvgPool2d(1)
    flatten: Flatten()
    dropout: Dropout(0.3)
    head: Linear(2048 -> 10)

train Experiment:
    model: ImageClassifier()
    data: CIFAR10()
    optimizer: AdamW(lr=3e-4)
    loss: CrossEntropy
    epochs: 20
```

</td>
<td width="50%" valign="top">

### Generated Python — 80+ lines

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
# ... more imports ...

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(
            weights='IMAGENET1K_V1'
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

# ... training loop, data loading,
#     metrics, checkpointing ...
```

</td>
</tr>
</table>

> **One file. Three backends.** The same `.axon` source compiles to PyTorch, TensorFlow, or JAX with a single flag.

<br>

## 🧠 All 35 Block Types

Axon covers the **entire ML ecosystem** — from data loading to distributed training to production serving.

<br>

<details open>
<summary><h3>🏗️ Core ML</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `model` | Define neural network architectures with arrow notation | `nn.Module` / `keras.Model` / `flax.linen` |
| `data` | Data loading, splits, transforms, streaming | `torchvision` · `tf.data` · `HuggingFace datasets` |
| `train` | Training loops — AMP, EMA, DDP, gradient accumulation | `PyTorch` · `mixed precision` · `distributed` |
| `evaluate` | Metrics, classification reports, confusion matrices | `sklearn` · `torchmetrics` · `numpy` |
| `search` | Hyperparameter optimization — Bayesian, TPE, CMA-ES | `Optuna` |
| `deploy` | Export, quantize, containerize, serve | `ONNX` · `TorchScript` · `Docker` |

</details>

<details>
<summary><h3>🎓 Advanced Training</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `finetune` | Parameter-efficient fine-tuning — LoRA, QLoRA, adapters | `HuggingFace PEFT` · `bitsandbytes` |
| `pretrain` | Self-supervised pretraining — MLM, contrastive, autoregressive | `transformers` · custom objectives |
| `distill` | Knowledge distillation — logit, attention, feature matching | Custom KD pipelines |
| `ensemble` | Model combination — voting, stacking, meta-learners | `sklearn` · custom |

</details>

<details>
<summary><h3>🎨 Generative AI</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `gan` | GAN training — Vanilla, WGAN, WGAN-GP, LSGAN, Hinge | Custom training loops |
| `diffusion` | Diffusion models — DDPM, DDIM, Euler, DPM-Solver | `diffusers` |
| `rag` | Retrieval-augmented generation — indexing + retrieval + gen | `LangChain` · `FAISS` · `ChromaDB` |
| `agent` | LLM agents with tools, memory, planning | `LangChain` · `LlamaIndex` |

</details>

<details>
<summary><h3>🌐 Domain-Specific</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `tabular` | Classical ML — classification, regression, ranking | `XGBoost` · `LightGBM` · `CatBoost` · `sklearn` |
| `timeseries` | Forecasting, anomaly detection, changepoint detection | `Prophet` · `pytorch-forecasting` · `statsmodels` |
| `graph` | Graph neural networks — node/edge/graph classification | `PyG` — `GCN` · `GAT` · `GraphSAGE` · `GIN` |
| `audio` | Speech recognition, TTS, audio classification | `torchaudio` · `Whisper` · `Wav2Vec2` |
| `multimodal` | Vision-language models, cross-modal fusion | `CLIP` · `BLIP` · cross-attention |
| `rl` | Reinforcement learning — policy + value methods | `stable-baselines3` — `PPO` · `DQN` · `SAC` · `TD3` |

</details>

<details>
<summary><h3>⚙️ MLOps & Production</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `monitor` | Experiment tracking, drift detection, alerting | `wandb` · `mlflow` · `tensorboard` · `neptune` |
| `serve` | Model serving APIs with batching, CORS, health checks | `FastAPI` · `Gradio` · `Streamlit` · `Triton` |
| `quantize` | Model compression — dynamic, static, QAT, GPTQ, AWQ | `torch.quantization` · `bitsandbytes` |
| `benchmark` | Performance profiling — latency, throughput, memory, FLOPs | Custom profilers |
| `test` | Model validation — invariance, robustness, bias | Custom test suites |

</details>

<details>
<summary><h3>🔧 Data, Features & Infrastructure</h3></summary>

| Block | What it does | Powered by |
|:------|:-------------|:-----------|
| `augment` | Augmentation pipelines for image, audio, text | `albumentations` · `audiomentations` · `nlpaug` |
| `feature` | Feature engineering — scaling, encoding, selection | `pandas` · `sklearn` transformers |
| `tokenizer` | Tokenizer training — BPE, WordPiece, Unigram | `tokenizers` · `sentencepiece` |
| `embedding` | Embedding extraction and indexing | `sentence-transformers` |
| `pipeline` | Multi-step ML workflows — sequential or parallel | Built-in orchestration |
| `callback` | Custom training hooks — epoch/batch triggers | Custom callbacks |
| `metric` | Custom metric definitions | `torchmetrics`-style |
| `federated` | Federated learning — privacy-preserving distributed ML | `Flower` — `FedAvg` · `FedProx` · `SCAFFOLD` |
| `automl` | Automated model selection and tuning | `AutoGluon` · `FLAML` · `H2O` · `TPOT` |
| `explain` | Model interpretability and feature importance | `SHAP` · `LIME` · `GradCAM` · attention viz |

</details>

<br>

## 📖 Syntax Showcase

<details open>
<summary><strong>Distributed Training with All the Bells and Whistles</strong></summary>

```axon
train LargeScaleExperiment:
    model: Transformer()
    data: WikiText()

    optimizer: AdamW(lr=6e-4, weight_decay=0.1)
    scheduler: CosineAnnealing(T_max=100)
    loss: CrossEntropy

    epochs: 100
    device: auto
    precision: bf16
    distributed: fsdp
    num_gpus: 8

    gradient_accumulation: 4
    gradient_clip_norm: 1.0
    ema: true
    ema_decay: 0.9999
    compile: true

    metrics:
        perplexity
        accuracy
```

</details>

<details>
<summary><strong>GAN Training</strong></summary>

```axon
gan StyleGAN:
    generator: Generator()
    discriminator: Discriminator()
    latent_dim: 512
    loss: wgan_gp
    gp_weight: 10.0
    n_critic: 5
    optimizer_g: Adam(lr=1e-4, betas=(0.0, 0.99))
    optimizer_d: Adam(lr=4e-4, betas=(0.0, 0.99))
    epochs: 200
```

</details>

<details>
<summary><strong>RAG Pipeline</strong></summary>

```axon
rag KnowledgeBase:
    retriever: faiss
    generator: "gpt-4"
    embedding_model: "all-MiniLM-L6-v2"
    chunk_size: 512
    chunk_overlap: 50
    top_k: 5
```

</details>

<details>
<summary><strong>Reinforcement Learning</strong></summary>

```axon
rl CartPoleSolver:
    algorithm: ppo
    environment: "CartPole-v1"
    policy: MlpPolicy
    total_timesteps: 500000
    learning_rate: 3e-4
    gamma: 0.99
    n_steps: 2048
    batch_size: 64
```

</details>

<details>
<summary><strong>LLM Agent with Tools</strong></summary>

```axon
agent ResearchAssistant:
    llm: "openai/gpt-4-turbo"
    tools: [web_search, code_interpreter, file_reader]
    memory: "conversation_buffer"
    max_iterations: 50
    system_prompt: "You are an expert researcher."
```

</details>

<details>
<summary><strong>Federated Learning</strong></summary>

```axon
federated HospitalNetwork:
    model: DiagnosisNet
    num_clients: 50
    rounds: 100
    strategy: fedavg
    fraction_fit: 0.3
```

</details>

<details>
<summary><strong>Full Production Pipeline</strong></summary>

```axon
pipeline Production:
    steps:
        - data_loading
        - feature_engineering
        - training
        - evaluation
        - optimization
        - deployment
        - monitoring
    parallel: false
```

</details>

<br>

## 📦 Supported Components

<table>
<tr>
<td width="50%" valign="top">

### 65+ Layers
`Conv1d` `Conv2d` `Conv3d` `ConvTranspose2d` `Linear` `Bilinear` `LazyLinear` `LSTM` `GRU` `RNN` `Transformer` `TransformerEncoder` `TransformerDecoder` `MultiheadAttention` `Embedding` `EmbeddingBag` `BatchNorm1d` `BatchNorm2d` `LayerNorm` `GroupNorm` `InstanceNorm` `RMSNorm` `Dropout` `ReLU` `GELU` `SiLU` `Mish` `LeakyReLU` `PReLU` `ELU` `SELU` `Tanh` `Sigmoid` `Softmax` `MaxPool2d` `AvgPool2d` `AdaptiveAvgPool2d` `Flatten` `Upsample` `PixelShuffle` and more

### 15 Optimizers
`Adam` `AdamW` `SGD` `RMSprop` `Adagrad` `Adadelta` `Adamax` `NAdam` `RAdam` `ASGD` `LBFGS` `SparseAdam` `Rprop` `Lion`

</td>
<td width="50%" valign="top">

### 60+ Pretrained Models
`ResNet-18/34/50/101/152` `EfficientNet-B0→B7` `EfficientNetV2` `VGG-11/13/16/19` `ViT-B/L/H` `Swin-T/S/B` `SwinV2` `DenseNet-121/161/169/201` `MobileNetV2/V3` `ConvNeXt` `RegNet` `InceptionV3` `MaxViT` and more

### 25+ Loss Functions
`CrossEntropy` `BCE` `BCEWithLogits` `MSE` `L1` `SmoothL1` `NLL` `KLDiv` `Huber` `TripletMargin` `CosineEmbedding` `FocalLoss` `DiceLoss` `ContrastiveLoss` `InfoNCE` and more

### 13 Schedulers
`CosineAnnealing` `WarmRestarts` `StepLR` `MultiStepLR` `ExponentialLR` `ReduceOnPlateau` `OneCycleLR` `LinearLR` `PolynomialLR` and more

</td>
</tr>
</table>

<br>

## 🏛️ Architecture

```
                              ┌─────────────────────────────────┐
                              │          source.axon            │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                ┌─────────────────────────────────────────────────────────┐
                │                    LEXER  (390 lines)                   │
                │  40+ keywords · indentation tracking · bracket depth    │
                │  strings · numbers · arrows · tensor types              │
                └───────────────────────────┬─────────────────────────────┘
                                            │  Token Stream
                                            ▼
                ┌─────────────────────────────────────────────────────────┐
                │                   PARSER  (960 lines)                   │
                │  Recursive descent · 35 block parsers · value exprs    │
                │  nested sub-blocks · multi-line brackets · error msgs   │
                └───────────────────────────┬─────────────────────────────┘
                                            │  Abstract Syntax Tree
                                            │  (739 lines, 58 node types)
                                            ▼
                ┌─────────────────────────────────────────────────────────┐
                │                 TRANSPILER  (4,860 lines)               │
                │  Multi-backend code gen · 50+ library integrations      │
                │  import management · cross-reference resolution         │
                └──────┬────────────────────┬────────────────────┬────────┘
                       │                    │                    │
                       ▼                    ▼                    ▼
                  ┌──────────┐        ┌──────────┐        ┌──────────┐
                  │ PyTorch  │        │TensorFlow│        │   JAX    │
                  │ nn.Module│        │  Keras   │        │  Flax    │
                  │ torchviz │        │  tf.data │        │  Optax   │
                  └──────────┘        └──────────┘        └──────────┘
```

<br>

## 📂 Project Structure

```
axon/
├── axon/                          # Core language implementation
│   ├── __init__.py                #   Version + public API
│   ├── __main__.py                #   python -m axon support
│   ├── parser/
│   │   ├── lexer.py               #   Multi-line string support, bracket-depth tracking
│   │   ├── parser.py              #   Recursive descent, 35 block types + plugin dispatch
│   │   └── ast_nodes.py           #   59 AST node types (incl. AxonImport)
│   ├── transpiler/
│   │   └── engine.py              #   Multi-backend code generation (PyTorch/TF/JAX)
│   ├── runtime/
│   │   ├── executor.py            #   Compile + execute orchestrator
│   │   └── config.py              #   Configuration management
│   ├── formatter.py               #   Code formatter (axon fmt)
│   ├── linter.py                  #   Static analysis — 13 rules (axon lint)
│   ├── semantic.py                #   Shape inference, ML pattern validation
│   ├── modules.py                 #   Module system with circular import detection
│   ├── plugins.py                 #   Plugin API with hooks and registry
│   ├── watcher.py                 #   File watcher for auto-recompile
│   ├── lsp/                       #   Language Server Protocol
│   │   ├── server.py              #     JSON-RPC 2.0 LSP server
│   │   ├── completions.py         #     Context-aware autocomplete
│   │   ├── diagnostics.py         #     Real-time error diagnostics
│   │   └── hover.py               #     Documentation on hover
│   └── types/
│       └── __init__.py            #   Type system placeholder
│
├── cli/
│   └── main.py                    #   CLI (compile/run/check/fmt/lint/watch/repl/lsp/init)
│
├── editors/vscode/                #   VS Code extension
│   ├── package.json               #     Extension manifest
│   ├── syntaxes/axon.tmLanguage.json  # TextMate syntax highlighting
│   ├── extension.js               #     LSP client
│   └── language-configuration.json
│
├── examples/                      #   19 comprehensive examples
│   ├── complete_vision.axon       #     Image classification end-to-end
│   ├── nlp_finetuning.axon        #     BERT fine-tuning with LoRA
│   ├── gan_image_generation.axon  #     WGAN-GP image synthesis
│   ├── diffusion_model.axon       #     DDPM with U-Net
│   ├── reinforcement_learning.axon#     PPO + DQN for game environments
│   ├── rag_pipeline.axon          #     Full RAG with FAISS + GPT-4
│   ├── llm_agent.axon             #     Multi-tool LLM agent
│   ├── distributed_training.axon  #     FSDP 8B parameter pretraining
│   ├── graph_neural_network.axon  #     GCN/GAT for molecular property prediction
│   ├── audio_speech.axon          #     Wav2Vec2 speech recognition
│   ├── multimodal_vlm.axon        #     CLIP-style vision-language model
│   ├── tabular_ml.axon            #     XGBoost + LightGBM ensemble
│   ├── time_series_forecasting.axon#    Transformer-based forecasting
│   ├── federated_learning.axon    #     Privacy-preserving hospital network
│   ├── model_optimization.axon    #     Quantization + distillation + pruning
│   ├── automl_pipeline.axon       #     AutoGluon automated ML
│   ├── generative_ai.axon         #     GPT-style text generation
│   └── end_to_end_production.axon #     Complete MLOps pipeline
│
├── tests/
│   └── test_axon.py               #   225 tests — lexer, parser, transpiler, tools, integration
│
├── QA_REPORT.md                   #   Comprehensive QA audit results
├── LANGUAGE_SPEC.md               #   Full language specification
├── pyproject.toml                 #   Package configuration
└── README.md
```

<br>

## 🧪 Tests

```bash
# Run the full test suite
pip install pytest
pytest tests/ -v
```

```
tests/test_axon.py::TestLexer::test_core_keywords           PASSED
tests/test_axon.py::TestLexer::test_extended_keywords        PASSED
tests/test_axon.py::TestLexer::test_arrow_notation           PASSED
tests/test_axon.py::TestParser::test_model_block             PASSED
tests/test_axon.py::TestParser::test_all_35_block_types      PASSED
tests/test_axon.py::TestTranspiler::test_pytorch_backend     PASSED
tests/test_axon.py::TestTranspiler::test_tensorflow_backend  PASSED
tests/test_axon.py::TestTranspiler::test_jax_backend         PASSED
tests/test_axon.py::TestIntegration::test_end_to_end         PASSED
...
========================= 153 passed in 0.23s =========================
```

All **18 examples** compile successfully across all 3 backends.

<br>

## 📥 Install

```bash
# From source
git clone https://github.com/desenyon/axon-lang.git
cd axon
pip install -e .

# With specific backend dependencies
pip install -e ".[pytorch]"     # PyTorch ecosystem
pip install -e ".[tensorflow]"  # TensorFlow ecosystem
pip install -e ".[jax]"         # JAX ecosystem
pip install -e ".[all]"         # Everything
```

<br>

## 🗺️ Roadmap

| Phase | Status | Focus |
|:------|:------:|:------|
| Core Language | ✅ | Lexer, parser, transpiler, 35 block types, 3 backends |
| QA & Hardening | ✅ | 225 tests, 18 examples, 12 bugs found and fixed |
| Multi-line Strings | ✅ | YAML-style `\|`, `\|>`, `\|-` block scalar syntax |
| `axon fmt` | ✅ | Code formatter with alignment, sorting, quote normalization |
| `axon lint` | ✅ | 13 lint rules (W001–W010, E001–E003) with `--fix` support |
| Watch Mode | ✅ | Auto-recompile on file changes with colored output |
| Semantic Analysis | ✅ | Shape inference, type checking, ML pattern validation |
| Module System | ✅ | `import Model from "./other.axon"` with circular import detection |
| Plugin API | ✅ | Custom block types, hooks, manifest-based discovery |
| LSP Server | ✅ | VS Code extension with autocomplete, hover, diagnostics, go-to-definition |

<br>

## 📄 License

MIT

<br>

---

<div align="center">

**Built with obsessive attention to the ML developer experience.**

*Axon v0.1.2*

</div>
