"""
Axon Static Linter
===================
Performs static analysis on Axon source code and returns
structured diagnostics.

Rules implemented:
  W001 – Unused block definitions
  W002 – Missing required fields
  W003 – Unknown layer / function names
  W004 – Inconsistent backend usage
  W005 – Large batch size (>512)
  W006 – Missing data split
  W007 – High learning rate (lr > 0.1)
  W008 – Deprecated patterns
  W009 – Naming conventions (PascalCase)
  W010 – Missing evaluation after training
  E001 – Duplicate block names
  E002 – Reference to undefined block
  E003 – Circular references
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any

from axon.parser.lexer import AxonLexer, TokenType


# ─── Severity ────────────────────────────────────────────────────

class Severity:
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ─── Diagnostic ──────────────────────────────────────────────────

@dataclass
class Diagnostic:
    code: str
    severity: str          # "error" | "warning" | "info"
    message: str
    line: int
    col: int = 0
    fixable: bool = False

    def __str__(self):
        return f"{self.severity.upper()} {self.code} (line {self.line}:{self.col}): {self.message}"


# Alias for backward compat
LintWarning = Diagnostic
LintError = Diagnostic


# ─── LintResult ──────────────────────────────────────────────────

class LintResult:
    def __init__(self, diagnostics: List[Diagnostic]):
        self._diagnostics = diagnostics

    @property
    def all(self) -> List[Diagnostic]:
        return self._diagnostics

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.INFO]

    def __len__(self):
        return len(self._diagnostics)

    def __bool__(self):
        return bool(self._diagnostics)

    def __iter__(self):
        return iter(self._diagnostics)

    def __repr__(self):
        return (f"LintResult({len(self.errors)} errors, "
                f"{len(self.warnings)} warnings, "
                f"{len(self.infos)} infos)")


# ─── Known ML layers ────────────────────────────────────────────

KNOWN_PYTORCH_LAYERS = {
    # Linear / MLP
    "Linear", "Bilinear", "LazyLinear",
    # Activations
    "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "GELU", "Sigmoid",
    "Tanh", "Softmax", "LogSoftmax", "Softplus", "Softsign", "Hardswish",
    "Mish", "SiLU", "GLU",
    # Conv
    "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
    "ConvTranspose3d", "LazyConv2d",
    # Pooling
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d",
    "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d", "GlobalAvgPool", "GlobalMaxPool",
    # Normalization
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "RMSNorm",
    # Dropout
    "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout",
    # Recurrent
    "RNN", "LSTM", "GRU", "LSTMCell", "GRUCell",
    # Attention / Transformer
    "MultiheadAttention", "TransformerEncoder", "TransformerDecoder",
    "TransformerEncoderLayer", "TransformerDecoderLayer", "Transformer",
    "Attention",
    # Embedding
    "Embedding", "EmbeddingBag",
    # Misc
    "Flatten", "Unflatten", "Upsample", "PixelShuffle", "Fold", "Unfold",
    "Identity", "Sequential", "ModuleList", "ModuleDict",
    "ResidualBlock", "BottleneckBlock", "InceptionBlock",
    # Common aliases used in Axon examples
    "CrossEntropy", "MSELoss", "BCELoss", "BCEWithLogitsLoss",
    "NLLLoss", "L1Loss", "SmoothL1Loss", "HuberLoss",
}

KNOWN_TF_LAYERS = {
    "Dense", "Conv2D", "MaxPooling2D", "AveragePooling2D", "Flatten",
    "LSTM", "GRU", "Dropout", "BatchNormalization", "LayerNormalization",
    "Embedding", "MultiHeadAttention", "GlobalAveragePooling2D",
    "GlobalMaxPooling2D", "Reshape", "Concatenate", "Add",
}

# Layers that are only in PyTorch (not TF)
PYTORCH_ONLY_LAYERS = {
    "Conv1d", "Conv3d", "BatchNorm1d", "BatchNorm3d", "LazyLinear",
    "LazyConv2d", "PixelShuffle", "RMSNorm", "Mish",
}

# Layers that are only in TF (not PyTorch)
TF_ONLY_LAYERS = {
    "Dense", "Conv2D", "MaxPooling2D", "BatchNormalization",
    "GlobalAveragePooling2D",
}

KNOWN_LAYERS = KNOWN_PYTORCH_LAYERS | KNOWN_TF_LAYERS

# Known non-layer names (optimizers, losses, schedulers, etc.) that are NOT block refs
KNOWN_NON_BLOCK_NAMES = {
    # Optimizers
    "Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax",
    "ASGD", "LBFGS", "Rprop", "NAdam", "RAdam",
    # Losses
    "CrossEntropy", "CrossEntropyLoss", "MSE", "MSELoss", "BCE",
    "BCELoss", "BCEWithLogitsLoss", "NLLLoss", "L1Loss", "SmoothL1Loss",
    "HuberLoss", "CTCLoss", "KLDivLoss", "PoissonNLLLoss", "TripletMarginLoss",
    "FocalLoss", "DiceLoss", "IoULoss",
    # Schedulers
    "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
    "ReduceLROnPlateau", "CyclicLR", "OneCycleLR", "WarmupScheduler",
    "CosineWarmup", "LinearWarmup",
    # Common functions / utilities used in values
    "True", "False", "None", "auto",
}

# Deprecated patterns → suggested replacement
DEPRECATED_PATTERNS = {
    "Adam(lr=": "Use Adam(lr=...) is fine; consider AdamW for regularisation",
    "batch_norm": "Use BatchNorm2d (PyTorch) or BatchNormalization (TF)",
    "relu": "Layer name 'relu' is fine but prefer ReLU() call syntax",
    "sigmoid": "Layer name 'sigmoid' is fine but prefer Sigmoid() call syntax",
}

# Block types that are expected to have certain fields
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "model": [],           # layers checked separately
    "data": ["source"],
    "train": ["optimizer", "loss", "epochs"],
    "evaluate": [],
    "search": ["method"],
    "deploy": [],
}

# Top-level block keywords
BLOCK_KEYWORDS = {
    "model", "data", "train", "evaluate", "search", "deploy",
    "pipeline", "transform", "pretrain", "finetune", "ensemble",
    "explain", "forward", "gan", "diffusion", "rl", "tabular",
    "timeseries", "graph", "audio", "multimodal", "distill",
    "quantize", "monitor", "serve", "test", "benchmark", "augment",
    "feature", "embedding", "tokenizer", "callback", "metric",
    "rag", "agent", "federated", "automl",
}


# ─── AxonLinter ──────────────────────────────────────────────────

class AxonLinter:
    """
    Perform static analysis on Axon source code.

    Usage::

        linter = AxonLinter(source)
        result = linter.lint()
        for d in result.errors:
            print(d)
    """

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()
        self._diagnostics: List[Diagnostic] = []

    def lint(self) -> LintResult:
        """Run all lint rules and return results."""
        self._diagnostics = []
        # Build a lightweight block model from lines
        blocks = self._parse_blocks()
        # Run all rules
        self._rule_E001_duplicate_names(blocks)
        self._rule_E002_undefined_references(blocks)
        self._rule_E003_circular_references(blocks)
        self._rule_W001_unused_blocks(blocks)
        self._rule_W002_missing_required_fields(blocks)
        self._rule_W003_unknown_layers(blocks)
        self._rule_W004_inconsistent_backend(blocks)
        self._rule_W005_large_batch_size(blocks)
        self._rule_W006_missing_data_split(blocks)
        self._rule_W007_high_learning_rate(blocks)
        self._rule_W008_deprecated_patterns(blocks)
        self._rule_W009_naming_conventions(blocks)
        self._rule_W010_missing_evaluation(blocks)
        return LintResult(self._diagnostics)

    def _emit(self, code: str, severity: str, message: str,
              line: int, col: int = 0, fixable: bool = False):
        self._diagnostics.append(
            Diagnostic(code=code, severity=severity, message=message,
                       line=line, col=col, fixable=fixable)
        )

    # ─── Block parser ──────────────────────────────────────────

    def _parse_blocks(self) -> List[Dict]:
        """
        Lightweight line-based block parser.
        Returns a list of dicts with keys:
          type, name, line, body_lines, props (dict of key → value string)
        """
        blocks = []
        current = None
        indent_level = 0

        for lineno, raw_line in enumerate(self.lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())

            # Detect block header
            m = re.match(
                r'^(' + '|'.join(sorted(BLOCK_KEYWORDS, key=len, reverse=True)) + r')\s+(\w+)\s*(?:\(.*\))?\s*:$',
                stripped
            )
            if m and indent == 0:
                current = {
                    "type": m.group(1),
                    "name": m.group(2),
                    "line": lineno,
                    "body_lines": [],
                    "props": {},
                    "raw_props": {},   # key → raw value string
                    "refs": [],        # referenced block names
                }
                blocks.append(current)
                indent_level = indent
                continue

            if current is not None and indent > 0:
                current["body_lines"].append((lineno, stripped))
                # Parse simple key: value
                kv = re.match(r'^(\w+)\s*:\s*(.+)$', stripped)
                if kv:
                    key = kv.group(1)
                    val = kv.group(2).strip()
                    current["props"][key] = val
                    current["raw_props"][key] = val
                    # Look for block references (Identifier() or Identifier.something)
                    refs = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\s*(?:\(|\.)|\b([A-Z][A-Za-z0-9_]*)\s*$', val)
                    for r in refs:
                        ref = r[0] or r[1]
                        if ref:
                            current["refs"].append(ref)

        return blocks

    def _get_block_map(self, blocks: List[Dict]) -> Dict[str, Dict]:
        """Map block name → block (last wins for duplicate detection)."""
        m = {}
        for b in blocks:
            m[b["name"]] = b
        return m

    # ─── Rules ─────────────────────────────────────────────────

    def _rule_E001_duplicate_names(self, blocks: List[Dict]):
        seen: Dict[str, int] = {}
        for b in blocks:
            name = b["name"]
            if name in seen:
                self._emit("E001", Severity.ERROR,
                           f"Duplicate block name '{name}' (first defined at line {seen[name]})",
                           b["line"])
            else:
                seen[name] = b["line"]

    def _rule_E002_undefined_references(self, blocks: List[Dict]):
        defined = {b["name"] for b in blocks}
        for b in blocks:
            for ref in b["refs"]:
                if (ref not in defined
                        and ref not in KNOWN_LAYERS
                        and ref not in KNOWN_NON_BLOCK_NAMES):
                    self._emit("E002", Severity.ERROR,
                               f"Block '{b['name']}' references undefined name '{ref}'",
                               b["line"])

    def _rule_E003_circular_references(self, blocks: List[Dict]):
        """Detect cycles in block references using DFS."""
        # Build adjacency: block name → set of referenced block names
        defined = {b["name"] for b in blocks}
        adj: Dict[str, Set[str]] = {}
        block_map = {b["name"]: b for b in blocks}
        for b in blocks:
            adj[b["name"]] = {r for r in b["refs"] if r in defined}

        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            in_stack.add(node)
            for neighbour in adj.get(node, set()):
                if neighbour not in visited:
                    result = dfs(neighbour, path + [neighbour])
                    if result is not None:
                        return result
                elif neighbour in in_stack:
                    return path + [neighbour]
            in_stack.discard(node)
            return None

        for b in blocks:
            if b["name"] not in visited:
                cycle = dfs(b["name"], [b["name"]])
                if cycle:
                    cycle_str = " → ".join(cycle)
                    self._emit("E003", Severity.ERROR,
                               f"Circular reference detected: {cycle_str}",
                               b["line"])

    def _rule_W001_unused_blocks(self, blocks: List[Dict]):
        """Warn about blocks defined but never referenced by any other block."""
        # Collect all referenced names
        all_refs: Set[str] = set()
        for b in blocks:
            for ref in b["refs"]:
                all_refs.add(ref)
        # Blocks that are entry points don't need to be referenced
        entry_types = {"train", "evaluate", "pipeline", "deploy", "serve",
                       "monitor", "test", "benchmark", "search", "rl",
                       "gan", "diffusion", "automl", "federated"}
        for b in blocks:
            if b["type"] in entry_types:
                continue  # entry points are not expected to be referenced
            if b["name"] not in all_refs:
                self._emit("W001", Severity.WARNING,
                           f"Block '{b['name']}' is defined but never referenced",
                           b["line"])

    def _rule_W002_missing_required_fields(self, blocks: List[Dict]):
        for b in blocks:
            btype = b["type"]
            required = REQUIRED_FIELDS.get(btype, [])
            for field_name in required:
                if field_name not in b["props"]:
                    self._emit("W002", Severity.WARNING,
                               f"'{btype}' block '{b['name']}' is missing required field '{field_name}'",
                               b["line"])
            # Model block: warn if no layers at all
            if btype == "model" and not b["body_lines"]:
                self._emit("W002", Severity.WARNING,
                           f"'model' block '{b['name']}' has no layers defined",
                           b["line"])

    def _rule_W003_unknown_layers(self, blocks: List[Dict]):
        """Warn if a layer name used in a model block is not a known layer."""
        layer_call_re = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\(')
        for b in blocks:
            if b["type"] != "model":
                continue
            for lineno, line_content in b["body_lines"]:
                matches = layer_call_re.findall(line_content)
                for layer in matches:
                    if layer not in KNOWN_LAYERS:
                        self._emit("W003", Severity.WARNING,
                                   f"Unknown layer or function '{layer}' in model '{b['name']}'",
                                   lineno)

    def _rule_W004_inconsistent_backend(self, blocks: List[Dict]):
        """Warn if PyTorch-only and TF-only layers are mixed."""
        for b in blocks:
            if b["type"] != "model":
                continue
            used_pytorch_only = set()
            used_tf_only = set()
            layer_call_re = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\(')
            for lineno, line_content in b["body_lines"]:
                for layer in layer_call_re.findall(line_content):
                    if layer in PYTORCH_ONLY_LAYERS:
                        used_pytorch_only.add(layer)
                    if layer in TF_ONLY_LAYERS:
                        used_tf_only.add(layer)
            if used_pytorch_only and used_tf_only:
                self._emit("W004", Severity.WARNING,
                           f"Model '{b['name']}' mixes PyTorch-only layers "
                           f"({', '.join(sorted(used_pytorch_only))}) "
                           f"with TF-only layers ({', '.join(sorted(used_tf_only))})",
                           b["line"])

    def _rule_W005_large_batch_size(self, blocks: List[Dict]):
        for b in blocks:
            for lineno, line_content in b["body_lines"]:
                m = re.match(r'batch_size\s*:\s*(\d+)', line_content)
                if m:
                    bs = int(m.group(1))
                    if bs > 512:
                        self._emit("W005", Severity.WARNING,
                                   f"Large batch_size={bs} in '{b['name']}' "
                                   f"(>512 may cause memory issues or poor generalisation)",
                                   lineno)

    def _rule_W006_missing_data_split(self, blocks: List[Dict]):
        for b in blocks:
            if b["type"] == "data":
                if "split" not in b["props"]:
                    self._emit("W006", Severity.WARNING,
                               f"Data block '{b['name']}' has no train/test split defined",
                               b["line"])

    def _rule_W007_high_learning_rate(self, blocks: List[Dict]):
        lr_re = re.compile(r'\blr\s*=\s*([\d.eE+-]+)')
        for b in blocks:
            for lineno, line_content in b["body_lines"]:
                m = lr_re.search(line_content)
                if m:
                    try:
                        lr = float(m.group(1))
                        if lr > 0.1:
                            self._emit("W007", Severity.WARNING,
                                       f"High learning rate lr={lr} in '{b['name']}' "
                                       f"(>0.1 may cause training instability)",
                                       lineno)
                    except ValueError:
                        pass

    def _rule_W008_deprecated_patterns(self, blocks: List[Dict]):
        for b in blocks:
            for lineno, line_content in b["body_lines"]:
                for pattern, suggestion in DEPRECATED_PATTERNS.items():
                    if pattern.lower() in line_content.lower():
                        self._emit("W008", Severity.INFO,
                                   f"Possibly deprecated pattern '{pattern.strip()}': {suggestion}",
                                   lineno, fixable=False)
                        break  # one warning per line

    def _rule_W009_naming_conventions(self, blocks: List[Dict]):
        """Block names should be PascalCase."""
        pascal_re = re.compile(r'^[A-Z][A-Za-z0-9]*$')
        for b in blocks:
            if not pascal_re.match(b["name"]):
                self._emit("W009", Severity.WARNING,
                           f"Block name '{b['name']}' should be PascalCase "
                           f"(e.g. '{b['name'].capitalize()}')",
                           b["line"], fixable=True)

    def _rule_W010_missing_evaluation(self, blocks: List[Dict]):
        """Warn if there's a train block but no evaluate block."""
        has_train = any(b["type"] == "train" for b in blocks)
        has_eval = any(b["type"] == "evaluate" for b in blocks)
        if has_train and not has_eval:
            for b in blocks:
                if b["type"] == "train":
                    self._emit("W010", Severity.WARNING,
                               f"Training block '{b['name']}' has no corresponding evaluate block",
                               b["line"])
                    break


def lint_source(source: str) -> LintResult:
    """Convenience function to lint Axon source code."""
    return AxonLinter(source).lint()
