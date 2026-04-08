"""
Axon LSP — Hover Provider
==========================
Returns documentation strings for block keywords, layers, optimizers,
loss functions, and other Axon constructs when the user hovers in an editor.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from axon.lsp.completions import (
    BLOCK_KEYWORDS,
    LAYER_COMPLETIONS,
    LOSS_COMPLETIONS,
    METRIC_COMPLETIONS,
    OPTIMIZER_COMPLETIONS,
)


# ---------------------------------------------------------------------------
# Extended block-type documentation
# ---------------------------------------------------------------------------

BLOCK_DOCS: Dict[str, str] = {
    "model": (
        "**model** — Neural Network Architecture\n\n"
        "Defines the layers and forward pass of a machine-learning model.\n\n"
        "```axon\n"
        "model MyNet:\n"
        "    fc1: Linear(784 -> 256)\n"
        "    relu: ReLU()\n"
        "    fc2: Linear(256 -> 10)\n"
        "```"
    ),
    "data": (
        "**data** — Dataset Definition\n\n"
        "Specifies data source, format, preprocessing transforms, and DataLoader settings.\n\n"
        "```axon\n"
        "data MyDataset:\n"
        "    source: \"./data\"\n"
        "    format: image_folder\n"
        "    split: 80/10/10\n"
        "    loader:\n"
        "        batch_size: 32\n"
        "        shuffle: true\n"
        "```"
    ),
    "train": (
        "**train** — Training Experiment\n\n"
        "Links a model and dataset, configures an optimizer, loss, and training schedule.\n\n"
        "```axon\n"
        "train MyExperiment:\n"
        "    model: MyNet()\n"
        "    data: MyDataset()\n"
        "    optimizer: Adam(lr=1e-3)\n"
        "    loss: CrossEntropy\n"
        "    epochs: 20\n"
        "```"
    ),
    "evaluate": (
        "**evaluate** — Evaluation Run\n\n"
        "Evaluates a model checkpoint on a test dataset.\n\n"
        "```axon\n"
        "evaluate MyExperiment:\n"
        "    checkpoint: \"best\"\n"
        "    data: MyDataset.test\n"
        "    metrics:\n"
        "        accuracy\n"
        "        f1\n"
        "```"
    ),
    "search": (
        "**search** — Hyperparameter Search\n\n"
        "Defines a hyperparameter optimization experiment.\n\n"
        "```axon\n"
        "search HPO:\n"
        "    model: MyNet()\n"
        "    data: MyDataset()\n"
        "    method: bayesian\n"
        "    metric: accuracy\n"
        "    trials: 50\n"
        "```"
    ),
    "deploy": (
        "**deploy** — Model Deployment\n\n"
        "Configures a model serving endpoint.\n\n"
        "```axon\n"
        "deploy API:\n"
        "    model: MyNet()\n"
        "    format: onnx\n"
        "    host: 0.0.0.0\n"
        "    port: 8080\n"
        "```"
    ),
    "pipeline": (
        "**pipeline** — End-to-End ML Pipeline\n\n"
        "Chains multiple Axon blocks into an automated workflow."
    ),
    "finetune": (
        "**finetune** — Fine-Tuning\n\n"
        "Adapts a pretrained model to a new task, with optional layer freezing and LoRA."
    ),
    "ensemble": "**ensemble** — Model Ensemble\n\nCombines predictions from multiple models.",
    "explain": "**explain** — Explainability\n\nGenerates feature-importance, SHAP, or attention visualisations.",
    "pretrain": "**pretrain** — Pretraining\n\nSelf-supervised or unsupervised pretraining block.",
    "gan": "**gan** — Generative Adversarial Network\n\nDefines generator and discriminator architectures.",
    "diffusion": "**diffusion** — Diffusion Model\n\nScore-based / DDPM-style generative model.",
    "rl": "**rl** — Reinforcement Learning\n\nEnvironment, policy, and reward setup.",
    "tabular": "**tabular** — Tabular Model\n\nGradient-boosted trees, MLP, or TabNet on structured data.",
    "timeseries": "**timeseries** — Time-Series Model\n\nForecast, anomaly detection, or classification on temporal data.",
    "graph": "**graph** — Graph Neural Network\n\nGCN, GAT, GraphSAGE, etc.",
    "audio": "**audio** — Audio Model\n\nSpeech, music, or environmental sound processing.",
    "multimodal": "**multimodal** — Multi-Modal Model\n\nFuses text, image, audio, or structured inputs.",
    "distill": "**distill** — Knowledge Distillation\n\nTransfers knowledge from a teacher to a smaller student model.",
    "quantize": "**quantize** — Model Quantization\n\nReduces model size and inference latency via INT8/INT4 quantization.",
    "monitor": "**monitor** — Monitoring\n\nLogs metrics, system resources, and model drift during training or serving.",
    "serve": "**serve** — Inference Server\n\nExposes a model as a REST or gRPC endpoint.",
    "test": "**test** — Test Block\n\nUnit or integration tests for Axon blocks.",
    "benchmark": "**benchmark** — Benchmarking\n\nMeasures throughput, latency, and memory usage.",
    "augment": "**augment** — Data Augmentation\n\nDefines on-the-fly augmentation transforms.",
    "feature": "**feature** — Feature Engineering\n\nDerives new features from raw inputs.",
    "embedding": "**embedding** — Embedding\n\nLookup table or pretrained embedding layer.",
    "tokenizer": "**tokenizer** — Tokenizer\n\nText tokenization (BPE, WordPiece, SentencePiece, etc.).",
    "callback": "**callback** — Training Callback\n\nCustom logic invoked at training lifecycle events.",
    "metric": "**metric** — Custom Metric\n\nUser-defined evaluation metric.",
    "rag": "**rag** — Retrieval-Augmented Generation\n\nCombines a retriever and a language model for QA.",
    "agent": "**agent** — AI Agent\n\nTool-using agent with planner and executor.",
    "federated": "**federated** — Federated Learning\n\nDistributed training across decentralised clients.",
    "automl": "**automl** — AutoML\n\nAutomated model selection, feature engineering, and tuning.",
}


# ---------------------------------------------------------------------------
# HoverProvider
# ---------------------------------------------------------------------------


class HoverProvider:
    """Returns Markdown hover content for a cursor position in an Axon file."""

    def get_hover(
        self,
        source: str,
        line: int,
        character: int,
        uri: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return an LSP Hover dict or ``None`` if nothing to show.

        Parameters
        ----------
        source:
            Full document source text.
        line:
            0-based line index of the cursor.
        character:
            0-based column of the cursor.
        uri:
            Document URI (reserved).
        """
        word = self._word_at(source, line, character)
        if not word:
            return None

        content = self._lookup(word)
        if not content:
            return None

        return {
            "contents": {
                "kind": "markdown",
                "value": content,
            }
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _lookup(self, word: str) -> Optional[str]:
        """Return Markdown documentation for *word*, or ``None``."""
        # Block keywords
        if word in BLOCK_DOCS:
            return BLOCK_DOCS[word]

        # Layers
        if word in LAYER_COMPLETIONS:
            sig = LAYER_COMPLETIONS[word]
            return f"**{word}** — Layer\n\n`{sig}`"

        # Optimizers
        if word in OPTIMIZER_COMPLETIONS:
            sig = OPTIMIZER_COMPLETIONS[word]
            return f"**{word}** — Optimizer\n\n`{sig}`"

        # Losses
        if word in LOSS_COMPLETIONS:
            sig = LOSS_COMPLETIONS[word]
            return f"**{word}** — Loss Function\n\n`{sig}`"

        # Metrics
        if word in METRIC_COMPLETIONS:
            desc = METRIC_COMPLETIONS[word]
            return f"**{word}** — Metric\n\n{desc}"

        # Plugin contributions
        try:
            from axon.plugins import PluginRegistry

            registry = PluginRegistry()
            all_completions = registry.get_all_completions()
            if word in all_completions:
                return f"**{word}** — Plugin-provided identifier"
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_at(source: str, line: int, character: int) -> Optional[str]:
        """Extract the identifier under the cursor."""
        lines = source.splitlines()
        if line >= len(lines):
            return None
        text = lines[line]
        if character > len(text):
            return None

        # Find word boundaries
        start = character
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1

        end = character
        while end < len(text) and (text[end].isalnum() or text[end] == "_"):
            end += 1

        word = text[start:end].strip()
        return word if word else None
