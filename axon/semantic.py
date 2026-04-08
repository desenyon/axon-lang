"""
Axon Semantic Analyzer
=======================
ML-aware deep semantic analysis of a parsed Axon AST (Program).

Analysis passes:
  1. Shape inference  – Track tensor shapes through Linear layers, detect mismatches
  2. Type checking    – Verify block references are correct types
  3. Reference resolution – Symbol table + verify all references resolve
  4. ML pattern validation – Missing activations, no normalization in deep models, etc.
  5. Cross-block validation – train blocks reference valid model/data blocks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set

from axon.parser.ast_nodes import (
    Program, ASTNode, ModelDef, DataDef, TrainDef, EvalDef,
    SearchDef, DeployDef, PipelineDef, FinetuneDef, EnsembleDef,
    ExplainDef, PretrainDef, GANDef, DiffusionDef, RLDef, TabularDef,
    TimeSeriesDef, GraphDef, AudioDef, MultimodalDef, DistillDef,
    QuantizeDef, MonitorDef, ServeDef, TestDef, BenchmarkDef,
    AugmentDef, FeatureDef, EmbeddingDef, TokenizerDef, CallbackDef,
    MetricDef, RAGDef, AgentDef, FederatedDef, AutoMLDef,
    FunctionCall, Identifier, NumberLiteral, StringLiteral,
    AttributeAccess, LayerDef, ArrowExpr,
)


# ─── Severity ───────────────────────────────────────────────────

class Severity:
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ─── Diagnostic ─────────────────────────────────────────────────

@dataclass
class SemanticDiagnostic:
    code: str
    severity: str
    message: str
    line: int
    col: int = 0

    def __str__(self):
        return (f"{self.severity.upper()} {self.code} "
                f"(line {self.line}:{self.col}): {self.message}")


# ─── SemanticResult ─────────────────────────────────────────────

class SemanticResult:
    def __init__(self, diagnostics: List[SemanticDiagnostic]):
        self._diagnostics = diagnostics

    @property
    def all(self) -> List[SemanticDiagnostic]:
        return self._diagnostics

    @property
    def errors(self) -> List[SemanticDiagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[SemanticDiagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.WARNING]

    @property
    def infos(self) -> List[SemanticDiagnostic]:
        return [d for d in self._diagnostics if d.severity == Severity.INFO]

    def __len__(self):
        return len(self._diagnostics)

    def __bool__(self):
        return bool(self._diagnostics)

    def __iter__(self):
        return iter(self._diagnostics)

    def __repr__(self):
        return (f"SemanticResult({len(self.errors)} errors, "
                f"{len(self.warnings)} warnings, "
                f"{len(self.infos)} infos)")


# ─── Block type constants ────────────────────────────────────────

MODEL_TYPES = (ModelDef,)
DATA_TYPES = (DataDef,)
TRAIN_TYPES = (TrainDef,)
EVAL_TYPES = (EvalDef,)


# ─── SemanticAnalyzer ────────────────────────────────────────────

class SemanticAnalyzer:
    """
    Perform ML-aware semantic analysis on a parsed Axon Program.

    Usage::

        from axon.parser.parser import AxonParser
        from axon.semantic import SemanticAnalyzer

        program = AxonParser(source).parse()
        analyzer = SemanticAnalyzer(program)
        result = analyzer.analyze()
        for d in result.errors:
            print(d)
    """

    def __init__(self, program: Program):
        self.program = program
        self._diagnostics: List[SemanticDiagnostic] = []
        # Symbol table: name → AST node
        self._symbols: Dict[str, ASTNode] = {}

    def analyze(self) -> SemanticResult:
        """Run all semantic analysis passes."""
        self._diagnostics = []
        self._symbols = {}

        # Pass 1: Build symbol table
        self._build_symbol_table()

        # Pass 2: Reference resolution
        self._pass_reference_resolution()

        # Pass 3: Type checking
        self._pass_type_checking()

        # Pass 4: Shape inference
        self._pass_shape_inference()

        # Pass 5: ML pattern validation
        self._pass_ml_patterns()

        # Pass 6: Cross-block validation
        self._pass_cross_block_validation()

        return SemanticResult(self._diagnostics)

    def _emit(self, code: str, severity: str, message: str,
              line: int = 0, col: int = 0):
        self._diagnostics.append(
            SemanticDiagnostic(code=code, severity=severity,
                               message=message, line=line, col=col)
        )

    # ─── Symbol Table ──────────────────────────────────────────

    def _build_symbol_table(self):
        seen_names: Dict[str, int] = {}
        for defn in self.program.definitions:
            name = getattr(defn, "name", None)
            if name is None:
                continue
            if name in seen_names:
                self._emit("S-E001", Severity.ERROR,
                           f"Duplicate definition '{name}' "
                           f"(first at line {seen_names[name]})",
                           getattr(defn, "line", 0))
            else:
                seen_names[name] = getattr(defn, "line", 0)
                self._symbols[name] = defn

    def _type_name(self, node: ASTNode) -> str:
        """Human-readable type name for a block node."""
        type_map = {
            ModelDef: "model", DataDef: "data", TrainDef: "train",
            EvalDef: "evaluate", SearchDef: "search", DeployDef: "deploy",
            PipelineDef: "pipeline", FinetuneDef: "finetune",
            EnsembleDef: "ensemble", ExplainDef: "explain",
            PretrainDef: "pretrain", GANDef: "gan",
            DiffusionDef: "diffusion", RLDef: "rl",
            TabularDef: "tabular", TimeSeriesDef: "timeseries",
            GraphDef: "graph", AudioDef: "audio",
            MultimodalDef: "multimodal", DistillDef: "distill",
            QuantizeDef: "quantize", MonitorDef: "monitor",
            ServeDef: "serve", TestDef: "test",
            BenchmarkDef: "benchmark", AugmentDef: "augment",
            FeatureDef: "feature", EmbeddingDef: "embedding",
            TokenizerDef: "tokenizer", CallbackDef: "callback",
            MetricDef: "metric", RAGDef: "rag",
            AgentDef: "agent", FederatedDef: "federated",
            AutoMLDef: "automl",
        }
        return type_map.get(type(node), type(node).__name__)

    # ─── Pass 2: Reference Resolution ──────────────────────────

    def _resolve_ref(self, ref: Any) -> Optional[str]:
        """Extract referenced block name from a reference value."""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, Identifier):
            return ref.name
        if isinstance(ref, FunctionCall):
            return ref.name
        if isinstance(ref, AttributeAccess):
            # e.g. MNISTData.test → MNISTData
            obj = ref.obj
            if isinstance(obj, Identifier):
                return obj.name
            if isinstance(obj, FunctionCall):
                return obj.name
        return None

    def _pass_reference_resolution(self):
        """Verify all block references resolve to defined names."""
        for defn in self.program.definitions:
            line = getattr(defn, "line", 0)

            # Train: model_ref, data_ref
            if isinstance(defn, TrainDef):
                for attr, field_name in [("model_ref", "model"), ("data_ref", "data")]:
                    ref = getattr(defn, attr, None)
                    if ref is None:
                        continue
                    name = self._resolve_ref(ref)
                    if name and name not in self._symbols:
                        self._emit("S-E002", Severity.ERROR,
                                   f"Train '{defn.name}': '{field_name}' references "
                                   f"undefined block '{name}'",
                                   line)

            # Evaluate: data_ref
            if isinstance(defn, EvalDef):
                ref = getattr(defn, "data_ref", None)
                if ref is not None:
                    name = self._resolve_ref(ref)
                    if name and name not in self._symbols:
                        self._emit("S-E002", Severity.ERROR,
                                   f"Evaluate '{defn.name}': 'data' references "
                                   f"undefined block '{name}'",
                                   line)

            # Ensemble: models list
            if isinstance(defn, EnsembleDef):
                for ref in defn.models:
                    name = self._resolve_ref(ref)
                    if name and name not in self._symbols:
                        self._emit("S-E002", Severity.ERROR,
                                   f"Ensemble '{defn.name}': references "
                                   f"undefined model '{name}'",
                                   line)

            # Distill: teacher, student
            if isinstance(defn, DistillDef):
                for attr in ["teacher", "student"]:
                    ref = getattr(defn, attr, "")
                    if ref and ref not in self._symbols:
                        self._emit("S-E002", Severity.ERROR,
                                   f"Distill '{defn.name}': '{attr}' references "
                                   f"undefined block '{ref}'",
                                   line)

    # ─── Pass 3: Type Checking ─────────────────────────────────

    def _pass_type_checking(self):
        """Verify referenced blocks are of the correct type."""
        for defn in self.program.definitions:
            line = getattr(defn, "line", 0)

            if isinstance(defn, TrainDef):
                # model_ref must point to a ModelDef
                ref = getattr(defn, "model_ref", None)
                if ref is not None:
                    name = self._resolve_ref(ref)
                    if name and name in self._symbols:
                        sym = self._symbols[name]
                        if not isinstance(sym, ModelDef):
                            self._emit("S-E003", Severity.ERROR,
                                       f"Train '{defn.name}': 'model' field references "
                                       f"'{name}' which is a {self._type_name(sym)}, "
                                       f"not a model",
                                       line)
                # data_ref must point to a DataDef
                ref = getattr(defn, "data_ref", None)
                if ref is not None:
                    name = self._resolve_ref(ref)
                    if name and name in self._symbols:
                        sym = self._symbols[name]
                        if not isinstance(sym, DataDef):
                            self._emit("S-E003", Severity.ERROR,
                                       f"Train '{defn.name}': 'data' field references "
                                       f"'{name}' which is a {self._type_name(sym)}, "
                                       f"not a data block",
                                       line)

            if isinstance(defn, EvalDef):
                ref = getattr(defn, "data_ref", None)
                if ref is not None:
                    name = self._resolve_ref(ref)
                    if name and name in self._symbols:
                        sym = self._symbols[name]
                        if not isinstance(sym, DataDef):
                            self._emit("S-E003", Severity.ERROR,
                                       f"Evaluate '{defn.name}': 'data' references "
                                       f"'{name}' which is a {self._type_name(sym)}, "
                                       f"not a data block",
                                       line)

            if isinstance(defn, EnsembleDef):
                for ref in defn.models:
                    name = self._resolve_ref(ref)
                    if name and name in self._symbols:
                        sym = self._symbols[name]
                        if not isinstance(sym, ModelDef):
                            self._emit("S-E003", Severity.ERROR,
                                       f"Ensemble '{defn.name}': '{name}' is a "
                                       f"{self._type_name(sym)}, not a model",
                                       line)

    # ─── Pass 4: Shape Inference ───────────────────────────────

    def _infer_layer_shape(self, layer_expr: Any) -> Optional[Tuple[int, int]]:
        """
        Try to extract (in_features, out_features) from a layer definition.
        Handles Linear(784 -> 256) and similar arrow expressions.
        """
        if isinstance(layer_expr, LayerDef):
            if layer_expr.layer_type in ("Linear", "Dense", "Bilinear"):
                for arg in layer_expr.args:
                    if isinstance(arg, ArrowExpr):
                        left = arg.left
                        right = arg.right
                        if isinstance(left, NumberLiteral) and isinstance(right, NumberLiteral):
                            return (int(left.value), int(right.value))
        if isinstance(layer_expr, FunctionCall):
            if layer_expr.name in ("Linear", "Dense", "Bilinear"):
                for arg in layer_expr.args:
                    if isinstance(arg, ArrowExpr):
                        left = arg.left
                        right = arg.right
                        if isinstance(left, NumberLiteral) and isinstance(right, NumberLiteral):
                            return (int(left.value), int(right.value))
        return None

    def _pass_shape_inference(self):
        """Check tensor shape compatibility between consecutive Linear layers."""
        for defn in self.program.definitions:
            if not isinstance(defn, ModelDef):
                continue
            line = getattr(defn, "line", 0)

            # Collect layer shapes in order
            layer_shapes: List[Tuple[str, Optional[Tuple[int, int]]]] = []
            for layer_name, layer_expr in defn.layers.items():
                shape = self._infer_layer_shape(layer_expr)
                layer_shapes.append((layer_name, shape))

            # Check consecutive linear layers for shape mismatches
            prev_name = None
            prev_out = None
            for layer_name, shape in layer_shapes:
                if shape is not None:
                    in_feat, out_feat = shape
                    if prev_out is not None and in_feat != prev_out:
                        self._emit("S-W001", Severity.ERROR,
                                   f"Shape mismatch in model '{defn.name}': "
                                   f"layer '{layer_name}' expects input {in_feat} "
                                   f"but previous layer '{prev_name}' outputs {prev_out}",
                                   line)
                    prev_out = out_feat
                    prev_name = layer_name
                else:
                    # Non-linear layer breaks the shape chain (could be activation etc.)
                    # We continue tracking if the prev_out remains valid
                    pass

    # ─── Pass 5: ML Pattern Validation ────────────────────────

    def _pass_ml_patterns(self):
        for defn in self.program.definitions:
            if not isinstance(defn, ModelDef):
                continue
            line = getattr(defn, "line", 0)
            layers = defn.layers
            layer_names = list(layers.keys())
            layer_types = []
            for lname, lexpr in layers.items():
                if isinstance(lexpr, (LayerDef, FunctionCall)):
                    ltype = (lexpr.layer_type if isinstance(lexpr, LayerDef)
                             else lexpr.name)
                    layer_types.append(ltype)
                else:
                    layer_types.append(str(lexpr))

            activation_names = {
                "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "GELU",
                "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Mish", "SiLU",
                "Softplus", "Hardswish", "GLU",
            }
            linear_names = {"Linear", "Dense", "Bilinear", "Conv1d", "Conv2d",
                            "Conv3d", "ConvTranspose2d"}
            norm_names = {
                "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm",
                "GroupNorm", "InstanceNorm2d", "BatchNormalization",
                "RMSNorm",
            }

            # W-ML001: No activation between consecutive linear layers
            for i in range(len(layer_types) - 1):
                curr = layer_types[i]
                nxt = layer_types[i + 1]
                if curr in linear_names and nxt in linear_names:
                    self._emit("S-W002", Severity.WARNING,
                               f"Model '{defn.name}': no activation between "
                               f"'{layer_names[i]}' ({curr}) and "
                               f"'{layer_names[i+1]}' ({nxt}). "
                               f"Consider adding ReLU or similar.",
                               line)

            # W-ML002: No normalization in deep models (>5 layers)
            if len(layer_types) > 5:
                has_norm = any(lt in norm_names for lt in layer_types)
                if not has_norm:
                    self._emit("S-W003", Severity.WARNING,
                               f"Model '{defn.name}' has {len(layer_types)} layers "
                               f"but no normalization layer (BatchNorm/LayerNorm). "
                               f"Consider adding normalization for training stability.",
                               line)

        # W-ML003: Learning rate checks per optimizer
        for defn in self.program.definitions:
            if not isinstance(defn, TrainDef):
                continue
            line = getattr(defn, "line", 0)
            optimizer = defn.optimizer
            if optimizer is None:
                continue
            opt_name = ""
            lr_val = None
            if isinstance(optimizer, FunctionCall):
                opt_name = optimizer.name
                lr_val = optimizer.kwargs.get("lr") or optimizer.kwargs.get("learning_rate")
                if lr_val is None and optimizer.args:
                    # positional lr
                    first_arg = optimizer.args[0]
                    if isinstance(first_arg, NumberLiteral):
                        lr_val = first_arg.value
                if isinstance(lr_val, NumberLiteral):
                    lr_val = lr_val.value

            if lr_val is not None:
                try:
                    lr = float(lr_val)
                    # SGD: typical lr 0.01–0.1
                    if opt_name == "SGD" and (lr < 1e-4 or lr > 0.5):
                        self._emit("S-W004", Severity.WARNING,
                                   f"Train '{defn.name}': SGD learning rate {lr} "
                                   f"is outside the typical range [0.0001, 0.5]",
                                   line)
                    # Adam: typical lr 1e-4 to 1e-2
                    elif opt_name in ("Adam", "AdamW") and (lr < 1e-6 or lr > 0.01):
                        self._emit("S-W004", Severity.WARNING,
                                   f"Train '{defn.name}': {opt_name} learning rate {lr} "
                                   f"is outside the typical range [1e-6, 0.01]",
                                   line)
                except (ValueError, TypeError):
                    pass

        # W-ML004: Batch size vs model size mismatch
        for defn in self.program.definitions:
            if not isinstance(defn, TrainDef):
                continue
            line = getattr(defn, "line", 0)
            # Check data block for batch_size
            data_ref_name = self._resolve_ref(getattr(defn, "data_ref", None))
            if data_ref_name and data_ref_name in self._symbols:
                data_block = self._symbols[data_ref_name]
                if isinstance(data_block, DataDef):
                    bs = data_block.loader_config.get("batch_size")
                    if bs is not None:
                        try:
                            bs_val = int(bs) if not isinstance(bs, NumberLiteral) else int(bs.value)
                            model_ref_name = self._resolve_ref(getattr(defn, "model_ref", None))
                            if model_ref_name and model_ref_name in self._symbols:
                                model = self._symbols[model_ref_name]
                                if isinstance(model, ModelDef):
                                    n_layers = len(model.layers)
                                    if bs_val > 1024 and n_layers > 10:
                                        self._emit("S-W005", Severity.WARNING,
                                                   f"Train '{defn.name}': large batch_size={bs_val} "
                                                   f"with deep model ({n_layers} layers) "
                                                   f"may cause memory issues",
                                                   line)
                        except (ValueError, TypeError):
                            pass

    # ─── Pass 6: Cross-block Validation ────────────────────────

    def _resolve_ref(self, ref: Any) -> Optional[str]:
        """Extract referenced block name from a reference value."""
        if isinstance(ref, str):
            return ref if ref else None
        if isinstance(ref, Identifier):
            return ref.name
        if isinstance(ref, FunctionCall):
            return ref.name
        if isinstance(ref, AttributeAccess):
            obj = ref.obj
            if isinstance(obj, Identifier):
                return obj.name
            if isinstance(obj, FunctionCall):
                return obj.name
        return None

    def _pass_cross_block_validation(self):
        """Verify train/evaluate blocks reference valid model and data blocks."""
        for defn in self.program.definitions:
            line = getattr(defn, "line", 0)

            if isinstance(defn, TrainDef):
                # Must have an optimizer
                if defn.optimizer is None:
                    self._emit("S-W006", Severity.WARNING,
                               f"Train '{defn.name}' has no optimizer defined",
                               line)
                # Must have a loss
                if defn.loss is None:
                    self._emit("S-W007", Severity.WARNING,
                               f"Train '{defn.name}' has no loss function defined",
                               line)
                # Must reference a model
                if defn.model_ref is None:
                    self._emit("S-W008", Severity.WARNING,
                               f"Train '{defn.name}' does not reference a model block",
                               line)
                # Must reference data
                if defn.data_ref is None:
                    self._emit("S-W009", Severity.WARNING,
                               f"Train '{defn.name}' does not reference a data block",
                               line)

            if isinstance(defn, EnsembleDef):
                if not defn.models:
                    self._emit("S-W010", Severity.WARNING,
                               f"Ensemble '{defn.name}' has no models listed",
                               line)

            if isinstance(defn, DistillDef):
                if not defn.teacher:
                    self._emit("S-W011", Severity.WARNING,
                               f"Distill '{defn.name}' has no teacher model",
                               line)
                if not defn.student:
                    self._emit("S-W012", Severity.WARNING,
                               f"Distill '{defn.name}' has no student model",
                               line)
                # Verify teacher and student are model blocks
                for attr in ["teacher", "student"]:
                    ref_name = getattr(defn, attr, "")
                    if ref_name and ref_name in self._symbols:
                        sym = self._symbols[ref_name]
                        if not isinstance(sym, ModelDef):
                            self._emit("S-E004", Severity.ERROR,
                                       f"Distill '{defn.name}': '{attr}' references "
                                       f"'{ref_name}' which is a "
                                       f"{self._type_name(sym)}, not a model",
                                       line)


def analyze_source(source: str) -> SemanticResult:
    """Convenience: parse source and run semantic analysis."""
    from axon.parser.parser import AxonParser
    program = AxonParser(source).parse()
    return SemanticAnalyzer(program).analyze()
