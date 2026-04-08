"""
Axon LSP — Completion Provider
================================
ML-aware completions for the Axon language server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Block keywords with descriptions
# ---------------------------------------------------------------------------

BLOCK_KEYWORDS: Dict[str, str] = {
    "model": "Define a neural network architecture",
    "data": "Define a dataset with loading and preprocessing",
    "train": "Configure a training experiment",
    "evaluate": "Evaluate model performance on a dataset",
    "search": "Hyperparameter search configuration",
    "deploy": "Model deployment configuration",
    "pipeline": "End-to-end ML pipeline",
    "finetune": "Fine-tune a pretrained model",
    "ensemble": "Combine multiple models",
    "explain": "Explainability / interpretability analysis",
    "pretrain": "Pretraining configuration",
    "gan": "Generative Adversarial Network",
    "diffusion": "Diffusion model configuration",
    "rl": "Reinforcement learning setup",
    "tabular": "Tabular data model",
    "timeseries": "Time-series model",
    "graph": "Graph Neural Network",
    "audio": "Audio model configuration",
    "multimodal": "Multi-modal model",
    "distill": "Knowledge distillation",
    "quantize": "Model quantization",
    "monitor": "Training monitoring / logging",
    "serve": "Serving / inference endpoint",
    "test": "Unit / integration test block",
    "benchmark": "Performance benchmarking",
    "augment": "Data augmentation pipeline",
    "feature": "Feature engineering",
    "embedding": "Embedding layer / lookup",
    "tokenizer": "Text tokenizer configuration",
    "callback": "Training callback",
    "metric": "Custom metric definition",
    "rag": "Retrieval-Augmented Generation",
    "agent": "AI agent configuration",
    "federated": "Federated learning setup",
    "automl": "Automated machine learning",
}

# ---------------------------------------------------------------------------
# Layer completions (65+)
# ---------------------------------------------------------------------------

LAYER_COMPLETIONS: Dict[str, str] = {
    # Linear / Dense
    "Linear": "Linear(in_features, out_features, bias=True)",
    "Dense": "Dense(units, activation=None)",
    "LazyLinear": "LazyLinear(out_features, bias=True)",
    # Convolutional
    "Conv1d": "Conv1d(in_channels, out_channels, kernel_size)",
    "Conv2d": "Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)",
    "Conv3d": "Conv3d(in_channels, out_channels, kernel_size)",
    "ConvTranspose1d": "ConvTranspose1d(in_channels, out_channels, kernel_size)",
    "ConvTranspose2d": "ConvTranspose2d(in_channels, out_channels, kernel_size)",
    "ConvTranspose3d": "ConvTranspose3d(in_channels, out_channels, kernel_size)",
    "DepthwiseConv2d": "DepthwiseConv2d(in_channels, kernel_size)",
    "SeparableConv2d": "SeparableConv2d(in_channels, out_channels, kernel_size)",
    # Recurrent
    "LSTM": "LSTM(input_size, hidden_size, num_layers=1, batch_first=True)",
    "GRU": "GRU(input_size, hidden_size, num_layers=1, batch_first=True)",
    "RNN": "RNN(input_size, hidden_size, num_layers=1)",
    "LSTMCell": "LSTMCell(input_size, hidden_size)",
    "GRUCell": "GRUCell(input_size, hidden_size)",
    # Transformer
    "MultiheadAttention": "MultiheadAttention(embed_dim, num_heads, dropout=0.0)",
    "TransformerEncoder": "TransformerEncoder(encoder_layer, num_layers)",
    "TransformerDecoder": "TransformerDecoder(decoder_layer, num_layers)",
    "TransformerEncoderLayer": "TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)",
    "TransformerDecoderLayer": "TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)",
    "Transformer": "Transformer(d_model=512, nhead=8)",
    # Normalization
    "BatchNorm1d": "BatchNorm1d(num_features)",
    "BatchNorm2d": "BatchNorm2d(num_features)",
    "BatchNorm3d": "BatchNorm3d(num_features)",
    "LayerNorm": "LayerNorm(normalized_shape)",
    "GroupNorm": "GroupNorm(num_groups, num_channels)",
    "InstanceNorm1d": "InstanceNorm1d(num_features)",
    "InstanceNorm2d": "InstanceNorm2d(num_features)",
    "RMSNorm": "RMSNorm(normalized_shape)",
    # Pooling
    "MaxPool1d": "MaxPool1d(kernel_size, stride=None)",
    "MaxPool2d": "MaxPool2d(kernel_size, stride=None)",
    "MaxPool3d": "MaxPool3d(kernel_size, stride=None)",
    "AvgPool1d": "AvgPool1d(kernel_size, stride=None)",
    "AvgPool2d": "AvgPool2d(kernel_size, stride=None)",
    "AvgPool3d": "AvgPool3d(kernel_size, stride=None)",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool2d(output_size)",
    "GlobalAvgPool": "GlobalAvgPool()",
    # Activation
    "ReLU": "ReLU(inplace=False)",
    "LeakyReLU": "LeakyReLU(negative_slope=0.01)",
    "PReLU": "PReLU(num_parameters=1)",
    "ELU": "ELU(alpha=1.0)",
    "SELU": "SELU()",
    "GELU": "GELU()",
    "Sigmoid": "Sigmoid()",
    "Tanh": "Tanh()",
    "Softmax": "Softmax(dim=-1)",
    "LogSoftmax": "LogSoftmax(dim=-1)",
    "Mish": "Mish()",
    "SiLU": "SiLU()  # Swish",
    "Hardswish": "Hardswish()",
    # Dropout / Regularisation
    "Dropout": "Dropout(p=0.5)",
    "Dropout2d": "Dropout2d(p=0.5)",
    "AlphaDropout": "AlphaDropout(p=0.5)",
    "SpatialDropout": "SpatialDropout(p=0.5)",
    # Embedding
    "Embedding": "Embedding(num_embeddings, embedding_dim)",
    "EmbeddingBag": "EmbeddingBag(num_embeddings, embedding_dim, mode='mean')",
    # Misc
    "Flatten": "Flatten(start_dim=1)",
    "Unflatten": "Unflatten(dim, unflattened_size)",
    "Identity": "Identity()",
    "PixelShuffle": "PixelShuffle(upscale_factor)",
    "Upsample": "Upsample(size=None, scale_factor=None, mode='nearest')",
    "Reshape": "Reshape(shape)",
}

# ---------------------------------------------------------------------------
# Optimizer completions (15+)
# ---------------------------------------------------------------------------

OPTIMIZER_COMPLETIONS: Dict[str, str] = {
    "Adam": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)",
    "AdamW": "AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)",
    "SGD": "SGD(lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)",
    "RMSprop": "RMSprop(lr=1e-2, alpha=0.99, eps=1e-8, momentum=0)",
    "Adagrad": "Adagrad(lr=0.01, lr_decay=0, weight_decay=0)",
    "Adadelta": "Adadelta(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)",
    "Adamax": "Adamax(lr=2e-3, betas=(0.9, 0.999), eps=1e-8)",
    "NAdam": "NAdam(lr=2e-3, betas=(0.9, 0.999), weight_decay=0)",
    "RAdam": "RAdam(lr=1e-3, betas=(0.9, 0.999), weight_decay=0)",
    "LBFGS": "LBFGS(lr=1, max_iter=20, max_eval=None)",
    "Rprop": "Rprop(lr=0.01, etas=(0.5, 1.2))",
    "Lion": "Lion(lr=1e-4, betas=(0.9, 0.99), weight_decay=0)",
    "Shampoo": "Shampoo(lr=0.01, momentum=0, weight_decay=0)",
    "Prodigy": "Prodigy(lr=1.0, weight_decay=0)",
    "Sophia": "Sophia(lr=1e-4, betas=(0.965, 0.99), rho=0.04)",
}

# ---------------------------------------------------------------------------
# Loss function completions (25+)
# ---------------------------------------------------------------------------

LOSS_COMPLETIONS: Dict[str, str] = {
    "CrossEntropy": "CrossEntropyLoss(weight=None, reduction='mean')",
    "BCELoss": "BCELoss(weight=None, reduction='mean')",
    "BCEWithLogitsLoss": "BCEWithLogitsLoss(pos_weight=None, reduction='mean')",
    "MSELoss": "MSELoss(reduction='mean')",
    "L1Loss": "L1Loss(reduction='mean')",
    "SmoothL1Loss": "SmoothL1Loss(beta=1.0, reduction='mean')",
    "HuberLoss": "HuberLoss(delta=1.0, reduction='mean')",
    "NLLLoss": "NLLLoss(weight=None, reduction='mean')",
    "KLDivLoss": "KLDivLoss(reduction='batchmean')",
    "CosineEmbeddingLoss": "CosineEmbeddingLoss(margin=0.0, reduction='mean')",
    "TripletMarginLoss": "TripletMarginLoss(margin=1.0, p=2, reduction='mean')",
    "MarginRankingLoss": "MarginRankingLoss(margin=0.0, reduction='mean')",
    "HingeEmbeddingLoss": "HingeEmbeddingLoss(margin=1.0, reduction='mean')",
    "MultiLabelSoftMarginLoss": "MultiLabelSoftMarginLoss(weight=None, reduction='mean')",
    "MultiMarginLoss": "MultiMarginLoss(p=1, margin=1.0, reduction='mean')",
    "PoissonNLLLoss": "PoissonNLLLoss(log_input=True, reduction='mean')",
    "GaussianNLLLoss": "GaussianNLLLoss(eps=1e-6, reduction='mean')",
    "SoftMarginLoss": "SoftMarginLoss(reduction='mean')",
    "FocalLoss": "FocalLoss(alpha=0.25, gamma=2.0)",
    "DiceLoss": "DiceLoss(smooth=1.0)",
    "IoULoss": "IoULoss()",
    "CTC": "CTCLoss(blank=0, reduction='mean')",
    "ContrastiveLoss": "ContrastiveLoss(margin=1.0)",
    "InfoNCE": "InfoNCELoss(temperature=0.07)",
    "PerceptualLoss": "PerceptualLoss(feature_layers=['relu3_3'])",
    "WassersteinLoss": "WassersteinLoss()",
}

# ---------------------------------------------------------------------------
# Metric completions
# ---------------------------------------------------------------------------

METRIC_COMPLETIONS: Dict[str, str] = {
    "accuracy": "Fraction of correctly classified samples",
    "precision": "TP / (TP + FP)",
    "recall": "TP / (TP + FN)",
    "f1": "Harmonic mean of precision and recall",
    "auc": "Area Under the ROC Curve",
    "roc_auc": "ROC AUC score",
    "mse": "Mean Squared Error",
    "mae": "Mean Absolute Error",
    "rmse": "Root Mean Squared Error",
    "r2": "Coefficient of determination",
    "bleu": "BLEU score for text generation",
    "rouge": "ROUGE score for summarisation",
    "perplexity": "Language model perplexity",
    "map": "Mean Average Precision",
    "ndcg": "Normalised Discounted Cumulative Gain",
    "iou": "Intersection over Union (object detection)",
    "dice": "Dice coefficient (segmentation)",
    "confusion_matrix": "Full confusion matrix",
    "top5_accuracy": "Top-5 classification accuracy",
    "mrr": "Mean Reciprocal Rank",
}

# ---------------------------------------------------------------------------
# Block-specific property completions
# ---------------------------------------------------------------------------

BLOCK_PROPERTIES: Dict[str, List[str]] = {
    "model": [
        "layers",
        "forward",
        "input_shape",
        "output_shape",
        "dropout",
        "activation",
        "pretrained",
        "backbone",
    ],
    "data": [
        "source",
        "format",
        "split",
        "transform",
        "loader",
        "batch_size",
        "shuffle",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "augmentation",
        "cache",
        "streaming",
    ],
    "train": [
        "model",
        "data",
        "optimizer",
        "loss",
        "epochs",
        "device",
        "metrics",
        "callbacks",
        "scheduler",
        "gradient_clip",
        "mixed_precision",
        "accumulation_steps",
        "seed",
        "resume",
        "checkpoint",
        "log_interval",
        "eval_interval",
    ],
    "evaluate": [
        "checkpoint",
        "data",
        "metrics",
        "device",
        "batch_size",
        "output",
    ],
    "deploy": [
        "model",
        "format",
        "endpoint",
        "host",
        "port",
        "workers",
        "timeout",
        "authentication",
    ],
    "search": [
        "model",
        "data",
        "method",
        "metric",
        "direction",
        "trials",
        "timeout",
        "params",
    ],
    "finetune": [
        "base_model",
        "data",
        "optimizer",
        "loss",
        "epochs",
        "freeze_layers",
        "learning_rate",
        "lora_rank",
    ],
    "quantize": [
        "model",
        "method",
        "bits",
        "calibration_data",
        "backend",
    ],
    "distill": [
        "teacher",
        "student",
        "data",
        "temperature",
        "alpha",
        "optimizer",
        "epochs",
    ],
}


# ---------------------------------------------------------------------------
# CompletionProvider
# ---------------------------------------------------------------------------


class CompletionProvider:
    """Provides context-aware LSP completions for Axon files."""

    def __init__(self) -> None:
        self._plugin_completions: List[str] = []

    def load_plugin_completions(self) -> None:
        """Pull completions from any loaded plugins."""
        try:
            from axon.plugins import PluginRegistry

            registry = PluginRegistry()
            self._plugin_completions = registry.get_all_completions()
        except Exception:
            pass

    def get_completions(
        self,
        source: str,
        line: int,
        character: int,
        uri: str = "",
    ) -> List[Dict[str, Any]]:
        """Return a list of LSP CompletionItem dicts.

        Parameters
        ----------
        source:
            Full source text of the document.
        line:
            0-based line number of the cursor.
        character:
            0-based column of the cursor.
        uri:
            Document URI (unused, reserved for future use).
        """
        lines = source.splitlines()
        current_line = lines[line] if line < len(lines) else ""
        prefix = current_line[:character]

        context = self._detect_context(source, line, character)
        items: List[Dict[str, Any]] = []

        if context == "top_level" or context is None:
            # Suggest block keywords
            for kw, desc in BLOCK_KEYWORDS.items():
                items.append(self._make_item(kw, "keyword", desc, 1))
            # Plugin block types
            for pc in self._plugin_completions:
                items.append(self._make_item(pc, "keyword", "Plugin block type", 1))
        else:
            block_type = context

            # Block-specific properties
            for prop in BLOCK_PROPERTIES.get(block_type, []):
                items.append(self._make_item(prop, "property", f"{block_type} property", 2))

            # If inside model block → offer layers
            if block_type in ("model", "finetune", "distill"):
                for layer, sig in LAYER_COMPLETIONS.items():
                    items.append(self._make_item(layer, "class", sig, 3))

            # If inside train block → offer optimizers, losses, metrics
            if block_type in ("train", "search", "finetune"):
                for opt, sig in OPTIMIZER_COMPLETIONS.items():
                    items.append(self._make_item(opt, "class", sig, 4))
                for loss, sig in LOSS_COMPLETIONS.items():
                    items.append(self._make_item(loss, "class", sig, 5))
                for metric, desc in METRIC_COMPLETIONS.items():
                    items.append(self._make_item(metric, "value", desc, 6))

            # If inside evaluate/benchmark → metrics
            if block_type in ("evaluate", "benchmark", "monitor"):
                for metric, desc in METRIC_COMPLETIONS.items():
                    items.append(self._make_item(metric, "value", desc, 6))

        # Always add known identifiers from the file
        identifiers = self._extract_identifiers(source)
        for ident in identifiers:
            items.append(self._make_item(ident, "variable", "Defined in file", 9))

        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_context(self, source: str, line: int, character: int) -> Optional[str]:
        """Return the block type the cursor is inside, or None for top-level."""
        lines = source.splitlines()
        if not lines:
            return None
        block_keywords = set(BLOCK_KEYWORDS.keys())

        # Walk backwards from cursor line looking for a block opener
        current_indent = self._line_indent(lines[line] if line < len(lines) else "")

        for i in range(min(line, len(lines) - 1), -1, -1):
            l = lines[i]
            stripped = l.strip()
            if not stripped:
                continue
            indent = self._line_indent(l)
            if indent < current_indent or i == line:
                # Check if this line starts a block
                first_word = stripped.split()[0] if stripped.split() else ""
                if first_word in block_keywords:
                    return first_word
            if indent == 0 and i < line:
                first_word = stripped.split()[0] if stripped.split() else ""
                if first_word in block_keywords:
                    return first_word
                # We've hit an unindented non-block line — we're at top level
                break
        return None

    @staticmethod
    def _line_indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    @staticmethod
    def _extract_identifiers(source: str) -> List[str]:
        """Collect names defined in the source (block names and identifiers)."""
        import re

        names: List[str] = []
        # Match block definitions: keyword Name:
        for match in re.finditer(
            r"^(?:model|data|train|evaluate|search|deploy|pipeline|finetune|ensemble"
            r"|explain|pretrain|gan|diffusion|rl|tabular|timeseries|graph|audio"
            r"|multimodal|distill|quantize|monitor|serve|test|benchmark|augment"
            r"|feature|embedding|tokenizer|callback|metric|rag|agent|federated|automl"
            r")\s+(\w+)\s*:",
            source,
            re.MULTILINE,
        ):
            names.append(match.group(1))
        return list(dict.fromkeys(names))  # deduplicate preserving order

    @staticmethod
    def _make_item(
        label: str,
        kind_str: str,
        detail: str = "",
        sort_priority: int = 5,
    ) -> Dict[str, Any]:
        """Build an LSP CompletionItem dict."""
        # LSP CompletionItemKind numbers
        kind_map = {
            "keyword": 14,
            "property": 10,
            "class": 7,
            "value": 12,
            "variable": 6,
        }
        return {
            "label": label,
            "kind": kind_map.get(kind_str, 1),
            "detail": detail,
            "sortText": f"{sort_priority}_{label}",
            "insertText": label,
        }
