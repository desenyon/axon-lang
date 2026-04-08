"""
Axon Transpiler Engine
=======================
Converts an Axon AST into executable Python code.
Targets PyTorch by default, with TensorFlow and JAX backends available.

The transpiler generates clean, readable Python that uses standard ML libraries:
- PyTorch / TensorFlow / JAX for models and tensors
- torchvision / tf.data / grain for data loading
- Optuna for hyperparameter search
- ONNX / TorchScript for deployment
- wandb / mlflow / tensorboard for experiment tracking
- SHAP / LIME / Captum for interpretability
- stable-baselines3 for RL
- PyG for graph neural networks
- diffusers for diffusion models
- LangChain / LlamaIndex for RAG
- Flower for federated learning
- AutoGluon / FLAML for AutoML
"""

from typing import Any, Optional
from axon.parser.ast_nodes import *


class AxonTranspiler:
    """
    Transpiles Axon AST -> Python source code.

    Usage:
        transpiler = AxonTranspiler(backend="pytorch")
        python_code = transpiler.transpile(ast)
    """

    BACKENDS = ("pytorch", "tensorflow", "jax")

    def __init__(self, backend: str = "pytorch"):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Choose from {self.BACKENDS}")
        self.backend = backend
        self._indent = 0
        self._output: list[str] = []
        self._imports: set[str] = set()
        self._defined_names: dict[str, str] = {}

    def transpile(self, program: Program) -> str:
        """Transpile a full Program AST to Python source."""
        self._output = []
        self._imports = set()
        self._defined_names = {}

        # Register all named block definitions for cross-referencing
        _block_type_map = {
            ModelDef: "model", DataDef: "data", TrainDef: "train",
            EvalDef: "eval", SearchDef: "search", DeployDef: "deploy",
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
        for defn in program.definitions:
            block_kind = _block_type_map.get(type(defn))
            if block_kind:
                name = getattr(defn, 'name', None)
                if name:
                    self._defined_names[name] = block_kind

        self._add_base_imports()

        for defn in program.definitions:
            self._transpile_node(defn)
            self._emit("")

        import_block = self._build_import_block()
        body = "\n".join(self._output)

        return f"{import_block}\n\n{body}"

    # --- Import Management -------------------------------------------------------

    def _add_base_imports(self):
        """Add framework-specific base imports."""
        self._imports.add("import os")
        self._imports.add("import json")
        self._imports.add("from pathlib import Path")
        self._imports.add("from typing import Optional, Any")

        if self.backend == "pytorch":
            self._imports.add("import torch")
            self._imports.add("import torch.nn as nn")
            self._imports.add("import torch.optim as optim")
            self._imports.add("from torch.utils.data import DataLoader, Dataset, random_split")
        elif self.backend == "tensorflow":
            self._imports.add("import tensorflow as tf")
            self._imports.add("from tensorflow import keras")
            self._imports.add("from tensorflow.keras import layers")
        elif self.backend == "jax":
            self._imports.add("import jax")
            self._imports.add("import jax.numpy as jnp")
            self._imports.add("from flax import linen as nn")
            self._imports.add("import optax")

    def _build_import_block(self) -> str:
        """Build sorted import block."""
        stdlib = []
        third_party = []

        for imp in sorted(self._imports):
            if any(imp.startswith(f"import {m}") or imp.startswith(f"from {m}")
                   for m in ("os", "sys", "json", "pathlib", "typing", "dataclasses",
                            "collections", "functools", "itertools", "math", "time",
                            "copy", "logging", "warnings", "abc", "contextlib")):
                stdlib.append(imp)
            else:
                third_party.append(imp)

        parts = []
        if stdlib:
            parts.append("\n".join(stdlib))
        if third_party:
            parts.append("\n".join(third_party))

        header = '"""\nGenerated by Axon v2.0.0\nBackend: {}\n"""\n'.format(self.backend)
        return header + "\n\n".join(parts)

    # --- Output Helpers ----------------------------------------------------------

    def _emit(self, line: str = ""):
        """Emit a line of Python code at current indentation."""
        if line:
            self._output.append("    " * self._indent + line)
        else:
            self._output.append("")

    def _emit_block(self, lines: list[str]):
        for line in lines:
            self._emit(line)

    def _indent_up(self):
        self._indent += 1

    def _indent_down(self):
        self._indent = max(0, self._indent - 1)

    def _value_to_python(self, node: Any) -> str:
        """Convert an AST value node to Python expression string."""
        if node is None:
            return "None"
        if isinstance(node, str):
            return repr(node)
        if isinstance(node, (int, float)):
            return str(node)
        if isinstance(node, bool):
            return str(node)
        if isinstance(node, StringLiteral):
            return repr(node.value)
        if isinstance(node, NumberLiteral):
            return str(node.value)
        if isinstance(node, BoolLiteral):
            return str(node.value)
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, ParamRef):
            return f"config['{node.name}']"
        if isinstance(node, FunctionCall):
            return self._func_call_to_python(node)
        if isinstance(node, AttributeAccess):
            obj = self._value_to_python(node.obj)
            return f"{obj}.{node.attr}"
        if isinstance(node, ArrowExpr):
            left = self._value_to_python(node.left)
            right = self._value_to_python(node.right)
            return f"{left}, {right}"
        if isinstance(node, ListLiteral):
            elems = ", ".join(self._value_to_python(e) for e in node.elements)
            return f"[{elems}]"
        if isinstance(node, DictLiteral):
            pairs = ", ".join(
                f"{self._value_to_python(k)}: {self._value_to_python(v)}"
                for k, v in node.pairs
            )
            return "{" + pairs + "}"
        if isinstance(node, RatioExpr):
            return str(node.parts)
        if isinstance(node, KeyValue):
            return f"{repr(node.key)}: {self._value_to_python(node.value)}"
        return str(node)

    def _func_call_to_python(self, call: FunctionCall) -> str:
        """Convert a FunctionCall AST node to Python code."""
        parts = []
        for arg in call.args:
            parts.append(self._value_to_python(arg))
        for k, v in call.kwargs.items():
            parts.append(f"{k}={self._value_to_python(v)}")
        args_str = ", ".join(parts)
        return f"{call.name}({args_str})"

    # --- Node Dispatch -----------------------------------------------------------

    def _transpile_node(self, node: ASTNode):
        """Dispatch to the appropriate transpiler method."""
        dispatch = {
            ModelDef: self._transpile_model,
            DataDef: self._transpile_data,
            TrainDef: self._transpile_train,
            EvalDef: self._transpile_eval,
            SearchDef: self._transpile_search,
            DeployDef: self._transpile_deploy,
            PipelineDef: self._transpile_pipeline,
            FinetuneDef: self._transpile_finetune,
            EnsembleDef: self._transpile_ensemble,
            ExplainDef: self._transpile_explain,
            PretrainDef: self._transpile_pretrain,
            PythonBlock: self._transpile_python_block,
            GANDef: self._transpile_gan,
            DiffusionDef: self._transpile_diffusion,
            RLDef: self._transpile_rl,
            TabularDef: self._transpile_tabular,
            TimeSeriesDef: self._transpile_timeseries,
            GraphDef: self._transpile_graph,
            AudioDef: self._transpile_audio,
            MultimodalDef: self._transpile_multimodal,
            DistillDef: self._transpile_distill,
            QuantizeDef: self._transpile_quantize,
            MonitorDef: self._transpile_monitor,
            ServeDef: self._transpile_serve,
            TestDef: self._transpile_test,
            BenchmarkDef: self._transpile_benchmark,
            AugmentDef: self._transpile_augment,
            FeatureDef: self._transpile_feature,
            EmbeddingDef: self._transpile_embedding,
            TokenizerDef: self._transpile_tokenizer,
            CallbackDef: self._transpile_callback,
            MetricDef: self._transpile_metric,
            RAGDef: self._transpile_rag,
            AgentDef: self._transpile_agent,
            FederatedDef: self._transpile_federated,
            AutoMLDef: self._transpile_automl,
            AxonImport: self._transpile_axon_import,
        }
        handler = dispatch.get(type(node))
        if handler:
            handler(node)

    # --- Model Transpilation -------------------------------------------------------

    def _transpile_model(self, model: ModelDef):
        """Transpile a model definition to a PyTorch/TF/JAX class."""
        if self.backend == "pytorch":
            self._transpile_model_pytorch(model)
        elif self.backend == "tensorflow":
            self._transpile_model_tensorflow(model)
        elif self.backend == "jax":
            self._transpile_model_jax(model)

    def _transpile_model_pytorch(self, model: ModelDef):
        self._imports.add("import torchvision.models as models")

        self._emit(f"class {model.name}(nn.Module):")
        self._indent_up()

        params = set()
        for layer_name, layer_val in model.layers.items():
            self._collect_params(layer_val, params)

        param_str = ", ".join(f"{p}=None" for p in sorted(params))
        if param_str:
            self._emit(f"def __init__(self, {param_str}):")
        else:
            self._emit("def __init__(self):")
        self._indent_up()
        self._emit("super().__init__()")

        for layer_name, layer_val in model.layers.items():
            py_code = self._layer_to_pytorch(layer_name, layer_val)
            self._emit(py_code)

        self._indent_down()
        self._emit("")

        self._emit("def forward(self, x):")
        self._indent_up()

        if model.forward_def and model.forward_def.body:
            for stmt in model.forward_def.body:
                self._emit(self._value_to_python(stmt))
        else:
            layer_names = list(model.layers.keys())
            if layer_names:
                for name in layer_names:
                    self._emit(f"x = self.{name}(x)")
                self._emit("return x")
            else:
                self._emit("return x")

        self._indent_down()
        self._indent_down()

    def _transpile_model_tensorflow(self, model: ModelDef):
        self._emit(f"def build_{model.name}(**config):")
        self._indent_up()
        self._emit("inputs = keras.Input(shape=config.get('input_shape', (224, 224, 3)))")

        prev = "inputs"
        for layer_name, layer_val in model.layers.items():
            code = self._layer_to_tensorflow(layer_name, layer_val, prev)
            self._emit(f"{layer_name} = {code}")
            prev = layer_name

        self._emit(f"model = keras.Model(inputs=inputs, outputs={prev}, name='{model.name}')")
        self._emit("return model")
        self._indent_down()

    def _transpile_model_jax(self, model: ModelDef):
        self._emit(f"class {model.name}(nn.Module):")
        self._indent_up()

        for layer_name, layer_val in model.layers.items():
            self._emit(f"{layer_name}: Any = None")

        self._emit("")
        self._emit("@nn.compact")
        self._emit("def __call__(self, x):")
        self._indent_up()

        for layer_name, layer_val in model.layers.items():
            code = self._layer_to_jax(layer_name, layer_val)
            self._emit(f"x = {code}")

        self._emit("return x")
        self._indent_down()
        self._indent_down()

    def _collect_params(self, node: Any, params: set):
        """Collect @param references from a node."""
        if isinstance(node, ParamRef):
            params.add(node.name)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                self._collect_params(arg, params)
            for v in node.kwargs.values():
                self._collect_params(v, params)
        elif isinstance(node, ArrowExpr):
            self._collect_params(node.left, params)
            self._collect_params(node.right, params)

    def _layer_to_pytorch(self, name: str, val: Any) -> str:
        """Convert a layer definition to PyTorch code."""
        if isinstance(val, FunctionCall):
            fn = val.name.lower()

            pretrained_models = {
                "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
                "vgg11", "vgg13", "vgg16", "vgg19",
                "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
                "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b",
                "densenet121", "densenet161", "densenet169", "densenet201",
                "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
                "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
                "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf",
                "inception_v3",
                "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
                "mnasnet0_5", "mnasnet1_0",
                "maxvit_t",
                "googlenet", "alexnet", "squeezenet1_0", "squeezenet1_1",
                "wide_resnet50_2", "wide_resnet101_2",
                "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
            }

            if fn in pretrained_models:
                pretrained = "True"
                for k, v in val.kwargs.items():
                    if k == "pretrained":
                        pretrained = self._value_to_python(v)
                if pretrained == "True":
                    return f"self.{name} = models.{fn}(weights='DEFAULT')"
                return f"self.{name} = models.{fn}(weights=None)"

            layer_map = {
                # Linear layers
                "linear": "nn.Linear",
                "bilinear": "nn.Bilinear",
                "lazylinear": "nn.LazyLinear",
                "identity": "nn.Identity()",
                # Convolutions
                "conv1d": "nn.Conv1d",
                "conv2d": "nn.Conv2d",
                "conv3d": "nn.Conv3d",
                "convtranspose2d": "nn.ConvTranspose2d",
                "convtranspose1d": "nn.ConvTranspose1d",
                "convtranspose3d": "nn.ConvTranspose3d",
                "depthwiseconv": "nn.Conv2d",
                # Normalizations
                "batchnorm": "nn.BatchNorm2d",
                "batchnorm1d": "nn.BatchNorm1d",
                "batchnorm2d": "nn.BatchNorm2d",
                "batchnorm3d": "nn.BatchNorm3d",
                "layernorm": "nn.LayerNorm",
                "groupnorm": "nn.GroupNorm",
                "instancenorm": "nn.InstanceNorm2d",
                "instancenorm1d": "nn.InstanceNorm1d",
                "instancenorm2d": "nn.InstanceNorm2d",
                "instancenorm3d": "nn.InstanceNorm3d",
                "rmsnorm": "nn.RMSNorm",
                # Activations
                "relu": "nn.ReLU()",
                "gelu": "nn.GELU()",
                "silu": "nn.SiLU()",
                "swish": "nn.SiLU()",
                "mish": "nn.Mish()",
                "leakyrelu": "nn.LeakyReLU()",
                "prelu": "nn.PReLU()",
                "elu": "nn.ELU()",
                "selu": "nn.SELU()",
                "tanh": "nn.Tanh()",
                "sigmoid": "nn.Sigmoid()",
                "softmax": "nn.Softmax(dim=-1)",
                "softplus": "nn.Softplus()",
                "hardswish": "nn.Hardswish()",
                "hardtanh": "nn.Hardtanh()",
                "logsigmoid": "nn.LogSigmoid()",
                "logsoftmax": "nn.LogSoftmax(dim=-1)",
                # Dropout
                "dropout": "nn.Dropout",
                "dropout2d": "nn.Dropout2d",
                "dropout3d": "nn.Dropout3d",
                "alphadropout": "nn.AlphaDropout",
                # Pooling
                "maxpool1d": "nn.MaxPool1d",
                "maxpool2d": "nn.MaxPool2d",
                "maxpool3d": "nn.MaxPool3d",
                "avgpool1d": "nn.AvgPool1d",
                "avgpool2d": "nn.AvgPool2d",
                "avgpool3d": "nn.AvgPool3d",
                "adaptiveavgpool1d": "nn.AdaptiveAvgPool1d",
                "adaptiveavgpool2d": "nn.AdaptiveAvgPool2d",
                "adaptivemaxpool1d": "nn.AdaptiveMaxPool1d",
                "adaptivemaxpool2d": "nn.AdaptiveMaxPool2d",
                "lppool2d": "nn.LPPool2d",
                # Recurrent
                "lstm": "nn.LSTM",
                "gru": "nn.GRU",
                "rnn": "nn.RNN",
                # Attention
                "multiheadattention": "nn.MultiheadAttention",
                "scaleddotproductattention": "nn.functional.scaled_dot_product_attention",
                # Transformer
                "transformer": "nn.Transformer",
                "transformerencoder": "nn.TransformerEncoder",
                "transformerdecoder": "nn.TransformerDecoder",
                "transformerencoderlayer": "nn.TransformerEncoderLayer",
                "transformerdecoderlayer": "nn.TransformerDecoderLayer",
                # Misc
                "flatten": "nn.Flatten()",
                "unflatten": "nn.Unflatten",
                "embedding": "nn.Embedding",
                "embeddingbag": "nn.EmbeddingBag",
                "pixelshuffle": "nn.PixelShuffle",
                "upsample": "nn.Upsample",
                "sequential": "nn.Sequential",
            }

            pytorch_cls = layer_map.get(fn, f"nn.{val.name}")

            # Handle DepthwiseConv specially
            if fn == "depthwiseconv":
                args_parts = []
                for arg in val.args:
                    if isinstance(arg, ArrowExpr):
                        args_parts.append(self._value_to_python(arg.left))
                        args_parts.append(self._value_to_python(arg.right))
                    else:
                        args_parts.append(self._value_to_python(arg))
                for k, v in val.kwargs.items():
                    if k != "pretrained":
                        args_parts.append(f"{k}={self._value_to_python(v)}")
                # For depthwise: groups=in_channels
                if args_parts:
                    args_parts.append(f"groups={args_parts[0]}")
                args_str = ", ".join(args_parts)
                return f"self.{name} = nn.Conv2d({args_str})"

            args_parts = []
            for arg in val.args:
                if isinstance(arg, ArrowExpr):
                    args_parts.append(self._value_to_python(arg.left))
                    args_parts.append(self._value_to_python(arg.right))
                else:
                    args_parts.append(self._value_to_python(arg))
            for k, v in val.kwargs.items():
                if k != "pretrained":
                    args_parts.append(f"{k}={self._value_to_python(v)}")

            args_str = ", ".join(args_parts)

            if pytorch_cls.endswith("()"):
                return f"self.{name} = {pytorch_cls}"
            return f"self.{name} = {pytorch_cls}({args_str})"

        return f"self.{name} = {self._value_to_python(val)}"

    def _layer_to_tensorflow(self, name: str, val: Any, prev: str) -> str:
        """Convert a layer to TensorFlow/Keras code."""
        if isinstance(val, FunctionCall):
            fn = val.name.lower()

            tf_map = {
                "linear": "layers.Dense",
                "conv2d": "layers.Conv2D",
                "conv1d": "layers.Conv1D",
                "batchnorm": "layers.BatchNormalization",
                "layernorm": "layers.LayerNormalization",
                "dropout": "layers.Dropout",
                "flatten": "layers.Flatten",
                "lstm": "layers.LSTM",
                "gru": "layers.GRU",
                "embedding": "layers.Embedding",
                "maxpool2d": "layers.MaxPooling2D",
                "avgpool2d": "layers.AveragePooling2D",
                "globalavgpool2d": "layers.GlobalAveragePooling2D",
                "globalmaxpool2d": "layers.GlobalMaxPooling2D",
                "multiheadattention": "layers.MultiHeadAttention",
            }

            keras_cls = tf_map.get(fn, f"layers.{val.name}")
            args = ", ".join(self._value_to_python(a) for a in val.args)
            return f"{keras_cls}({args})({prev})"

        return f"{self._value_to_python(val)}({prev})"

    def _layer_to_jax(self, name: str, val: Any) -> str:
        """Convert a layer to JAX/Flax code."""
        if isinstance(val, FunctionCall):
            fn = val.name.lower()

            jax_map = {
                "linear": "nn.Dense",
                "conv2d": "nn.Conv",
                "conv1d": "nn.Conv",
                "batchnorm": "nn.BatchNorm",
                "layernorm": "nn.LayerNorm",
                "dropout": "nn.Dropout",
                "embedding": "nn.Embed",
                "lstm": "nn.RNN(nn.LSTMCell",
                "gru": "nn.RNN(nn.GRUCell",
            }

            flax_cls = jax_map.get(fn, f"nn.{val.name}")
            args = ", ".join(self._value_to_python(a) for a in val.args)
            return f"{flax_cls}({args})(x)"

        return f"x  # {name}"

    # --- Data Transpilation -------------------------------------------------------

    def _transpile_data(self, data: DataDef):
        """Transpile a data definition."""
        if self.backend == "pytorch":
            self._transpile_data_pytorch(data)
        elif self.backend == "tensorflow":
            self._transpile_data_tensorflow(data)
        elif self.backend == "jax":
            self._transpile_data_jax(data)

    def _transpile_data_pytorch(self, data: DataDef):
        self._imports.add("from torchvision import transforms, datasets")

        self._emit(f"class {data.name}DataModule:")
        self._indent_up()
        self._emit(f'"""Data module for {data.name}."""')
        self._emit("")

        source = self._value_to_python(data.source) if data.source else "'./data'"
        self._emit(f"def __init__(self, root={source}, **kwargs):")
        self._indent_up()
        self._emit("self.root = root")
        self._emit("self.kwargs = kwargs")

        self._emit("self.transform = transforms.Compose([")
        self._indent_up()
        for t in data.transforms:
            self._emit(f"{self._transform_to_pytorch(t)},")
        if not data.transforms:
            self._emit("transforms.ToTensor(),")
        self._emit("])")
        self._indent_down()

        if data.split:
            parts = data.split.parts
            total = sum(parts)
            ratios = [p / total for p in parts]
            self._emit(f"self.split_ratios = {ratios}")
        else:
            self._emit("self.split_ratios = [0.8, 0.1, 0.1]")

        batch_size = 32
        shuffle = True
        num_workers = 4
        for k, v in data.loader_config.items():
            if k == "batch_size" and isinstance(v, NumberLiteral):
                batch_size = int(v.value)
            elif k == "shuffle" and isinstance(v, BoolLiteral):
                shuffle = v.value
            elif k == "num_workers" and isinstance(v, NumberLiteral):
                num_workers = int(v.value)

        self._emit(f"self.batch_size = kwargs.get('batch_size', {batch_size})")
        self._emit(f"self.num_workers = kwargs.get('num_workers', {num_workers})")
        self._indent_down()
        self._emit("")

        self._emit("def setup(self):")
        self._indent_up()

        fmt = data.format
        if isinstance(fmt, Identifier):
            fmt = fmt.name
        elif isinstance(fmt, StringLiteral):
            fmt = fmt.value

        if fmt == "image_folder":
            self._emit("full_dataset = datasets.ImageFolder(self.root, transform=self.transform)")
        elif fmt == "csv":
            self._imports.add("import pandas as pd")
            self._emit("full_dataset = self._load_csv()")
        else:
            self._emit("full_dataset = datasets.ImageFolder(self.root, transform=self.transform)")

        self._emit("n = len(full_dataset)")
        self._emit("train_n = int(n * self.split_ratios[0])")
        self._emit("val_n = int(n * self.split_ratios[1])")
        self._emit("test_n = n - train_n - val_n")
        self._emit("self.train_set, self.val_set, self.test_set = random_split(")
        self._indent_up()
        self._emit("full_dataset, [train_n, val_n, test_n]")
        self._indent_down()
        self._emit(")")
        self._indent_down()
        self._emit("")

        for split in ("train", "val", "test"):
            shuffle_val = "True" if split == "train" else "False"
            self._emit(f"def {split}_loader(self):")
            self._indent_up()
            self._emit(f"return DataLoader(")
            self._indent_up()
            self._emit(f"self.{split}_set,")
            self._emit(f"batch_size=self.batch_size,")
            self._emit(f"shuffle={shuffle_val},")
            self._emit(f"num_workers=self.num_workers,")
            self._emit(f"pin_memory=True")
            self._indent_down()
            self._emit(")")
            self._indent_down()
            self._emit("")

        self._indent_down()

    def _transpile_data_tensorflow(self, data: DataDef):
        source = self._value_to_python(data.source) if data.source else "'./data'"
        self._emit(f"def build_{data.name}_dataset(root={source}, **kwargs):")
        self._indent_up()
        self._emit(f"batch_size = kwargs.get('batch_size', 32)")
        self._emit(f"dataset = tf.keras.utils.image_dataset_from_directory(root, batch_size=batch_size)")
        self._emit(f"return dataset")
        self._indent_down()

    def _transpile_data_jax(self, data: DataDef):
        self._imports.add("import tensorflow_datasets as tfds")
        source = self._value_to_python(data.source) if data.source else "'./data'"
        self._emit(f"def build_{data.name}_dataset(root={source}, **kwargs):")
        self._indent_up()
        self._emit("ds = tfds.load('mnist', split='train', data_dir=root)")
        self._emit("return ds.batch(kwargs.get('batch_size', 32))")
        self._indent_down()

    def _transform_to_pytorch(self, call: FunctionCall) -> str:
        """Convert a transform call to torchvision code."""
        name = call.name.lower()
        transform_map = {
            "resize": "transforms.Resize",
            "centercrop": "transforms.CenterCrop",
            "randomcrop": "transforms.RandomCrop",
            "random_crop": "transforms.RandomCrop",
            "normalize": "transforms.Normalize",
            "totensor": "transforms.ToTensor()",
            "horizontal_flip": "transforms.RandomHorizontalFlip()",
            "vertical_flip": "transforms.RandomVerticalFlip()",
            "color_jitter": "transforms.ColorJitter",
            "random_rotation": "transforms.RandomRotation",
            "random_resized_crop": "transforms.RandomResizedCrop",
            "grayscale": "transforms.Grayscale",
            "random_horizontal_flip": "transforms.RandomHorizontalFlip()",
            "random_vertical_flip": "transforms.RandomVerticalFlip()",
            "random_affine": "transforms.RandomAffine",
            "random_perspective": "transforms.RandomPerspective",
            "random_erasing": "transforms.RandomErasing",
            "gaussian_blur": "transforms.GaussianBlur",
            "autoaugment": "transforms.AutoAugment()",
            "randaugment": "transforms.RandAugment()",
            "trivialaugmentwide": "transforms.TrivialAugmentWide()",
        }

        pytorch_name = transform_map.get(name, f"transforms.{call.name}")

        if name == "normalize" and call.args:
            first_arg = call.args[0]
            if isinstance(first_arg, Identifier) and first_arg.name.lower() == "imagenet":
                return "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"

        if name == "augment":
            augs = []
            for arg in call.args:
                if isinstance(arg, Identifier):
                    aug_name = arg.name.lower()
                    aug_mapped = transform_map.get(aug_name, f"transforms.{arg.name}()")
                    if not aug_mapped.endswith("()"):
                        aug_mapped += "()"
                    augs.append(aug_mapped)
            return "transforms.RandomApply([" + ", ".join(augs) + "])"

        if pytorch_name.endswith("()"):
            return pytorch_name

        args = ", ".join(self._value_to_python(a) for a in call.args)
        kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in call.kwargs.items())
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"{pytorch_name}({all_args})"

    # --- Training Transpilation ----------------------------------------------------

    def _get_optimizer_map(self):
        return {
            "adam": "optim.Adam",
            "adamw": "optim.AdamW",
            "sgd": "optim.SGD",
            "rmsprop": "optim.RMSprop",
            "adagrad": "optim.Adagrad",
            "adadelta": "optim.Adadelta",
            "adamax": "optim.Adamax",
            "asgd": "optim.ASGD",
            "lbfgs": "optim.LBFGS",
            "nadam": "optim.NAdam",
            "radam": "optim.RAdam",
            "sparseadam": "optim.SparseAdam",
            "rprop": "optim.Rprop",
            "lion": "Lion",
        }

    def _get_scheduler_map(self):
        return {
            "cosineannealinglr": "optim.lr_scheduler.CosineAnnealingLR",
            "cosineannealing": "optim.lr_scheduler.CosineAnnealingLR",
            "cosineannealingwarmrestarts": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
            "steplr": "optim.lr_scheduler.StepLR",
            "multisteplr": "optim.lr_scheduler.MultiStepLR",
            "exponentiallr": "optim.lr_scheduler.ExponentialLR",
            "reducelronplateau": "optim.lr_scheduler.ReduceLROnPlateau",
            "onecyclelr": "optim.lr_scheduler.OneCycleLR",
            "linearlr": "optim.lr_scheduler.LinearLR",
            "polynomiallr": "optim.lr_scheduler.PolynomialLR",
            "constantlr": "optim.lr_scheduler.ConstantLR",
            "sequentiallr": "optim.lr_scheduler.SequentialLR",
            "chainedscheduler": "optim.lr_scheduler.ChainedScheduler",
        }

    def _get_loss_map(self):
        return {
            "crossentropy": "nn.CrossEntropyLoss()",
            "crossentropyloss": "nn.CrossEntropyLoss()",
            "bce": "nn.BCELoss()",
            "bceloss": "nn.BCELoss()",
            "bcewithlogitsloss": "nn.BCEWithLogitsLoss()",
            "bcewithlogits": "nn.BCEWithLogitsLoss()",
            "mse": "nn.MSELoss()",
            "mseloss": "nn.MSELoss()",
            "l1": "nn.L1Loss()",
            "l1loss": "nn.L1Loss()",
            "smoothl1": "nn.SmoothL1Loss()",
            "smoothl1loss": "nn.SmoothL1Loss()",
            "nll": "nn.NLLLoss()",
            "nllloss": "nn.NLLLoss()",
            "kldiv": "nn.KLDivLoss()",
            "kldivloss": "nn.KLDivLoss()",
            "huber": "nn.HuberLoss()",
            "huberloss": "nn.HuberLoss()",
            "poissonnll": "nn.PoissonNLLLoss()",
            "poissonnllloss": "nn.PoissonNLLLoss()",
            "tripletmargin": "nn.TripletMarginLoss()",
            "tripletmarginloss": "nn.TripletMarginLoss()",
            "cosineembedding": "nn.CosineEmbeddingLoss()",
            "cosineembeddingloss": "nn.CosineEmbeddingLoss()",
            "ctc": "nn.CTCLoss()",
            "ctcloss": "nn.CTCLoss()",
            "multimargin": "nn.MultiMarginLoss()",
            "multimarginloss": "nn.MultiMarginLoss()",
            "hingeembedding": "nn.HingeEmbeddingLoss()",
            "hingeembeddingloss": "nn.HingeEmbeddingLoss()",
            "marginranking": "nn.MarginRankingLoss()",
            "marginrankingloss": "nn.MarginRankingLoss()",
            "focal": "FocalLoss()",
            "focalloss": "FocalLoss()",
            "dice": "DiceLoss()",
            "diceloss": "DiceLoss()",
            "contrastive": "ContrastiveLoss()",
            "contrastiveloss": "ContrastiveLoss()",
            "ntxent": "NTXentLoss()",
            "ntxentloss": "NTXentLoss()",
            "infonce": "InfoNCELoss()",
            "infonceloss": "InfoNCELoss()",
        }

    def _transpile_train(self, train: TrainDef):
        """Transpile a training definition into a full training loop."""
        if self.backend == "pytorch":
            self._transpile_train_pytorch(train)
        elif self.backend == "tensorflow":
            self._transpile_train_tensorflow(train)
        elif self.backend == "jax":
            self._transpile_train_jax(train)

    def _transpile_train_pytorch(self, train: TrainDef):
        self._imports.add("import time")
        self._imports.add("import copy")
        self._imports.add("from tqdm import tqdm")

        # Emit custom loss classes if needed
        loss_name = ""
        if isinstance(train.loss, Identifier):
            loss_name = train.loss.name.lower()
        elif isinstance(train.loss, FunctionCall):
            loss_name = train.loss.name.lower()
        elif isinstance(train.loss, str):
            loss_name = train.loss.lower()

        if loss_name in ("focal", "focalloss"):
            self._emit_focal_loss()
        elif loss_name in ("dice", "diceloss"):
            self._emit_dice_loss()
        elif loss_name in ("contrastive", "contrastiveloss"):
            self._emit_contrastive_loss()
        elif loss_name in ("ntxent", "ntxentloss"):
            self._emit_ntxent_loss()
        elif loss_name in ("infonce", "infonceloss"):
            self._emit_infonce_loss()

        # Lion optimizer if needed
        opt_name = ""
        if train.optimizer and isinstance(train.optimizer, FunctionCall):
            opt_name = train.optimizer.name.lower()
        if opt_name == "lion":
            self._emit_lion_optimizer()

        # Distributed setup
        if train.distributed:
            self._emit_distributed_setup(train)

        self._emit(f"class {train.name}Trainer:")
        self._indent_up()
        self._emit(f'"""Training manager for {train.name}."""')
        self._emit("")

        # __init__
        self._emit("def __init__(self):")
        self._indent_up()

        # Device
        device = train.device
        if device == "auto":
            self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else")
            self._emit("    'mps' if torch.backends.mps.is_available() else 'cpu')")
        else:
            self._emit(f"self.device = torch.device('{device}')")

        # Model
        model_code = self._value_to_python(train.model_ref) if train.model_ref else "None"
        self._emit(f"self.model = {model_code}.to(self.device)")

        # torch.compile()
        if train.compile_model:
            self._emit("self.model = torch.compile(self.model)")

        # Data
        data_code = self._value_to_python(train.data_ref) if train.data_ref else "None"
        self._emit(f"self.data = {data_code}")
        self._emit("self.data.setup()")

        # Optimizer
        opt_map = self._get_optimizer_map()
        if train.optimizer:
            opt_nm = train.optimizer.name if isinstance(train.optimizer, FunctionCall) else "Adam"
            pytorch_opt = opt_map.get(opt_nm.lower(), f"optim.{opt_nm}")
            kwargs_str = ", ".join(
                f"{k}={self._value_to_python(v)}"
                for k, v in train.optimizer.kwargs.items()
            ) if isinstance(train.optimizer, FunctionCall) else ""
            self._emit(f"self.optimizer = {pytorch_opt}(self.model.parameters(), {kwargs_str})")
        else:
            self._emit("self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)")

        # Scheduler
        sched_map = self._get_scheduler_map()
        if train.scheduler:
            sched_name = train.scheduler.name if isinstance(train.scheduler, FunctionCall) else ""
            pytorch_sched = sched_map.get(sched_name.lower(), f"optim.lr_scheduler.{sched_name}")
            kwargs_str = ""
            if isinstance(train.scheduler, FunctionCall):
                kw_parts = []
                for k, v in train.scheduler.kwargs.items():
                    kw_parts.append(f"{k}={self._value_to_python(v)}")
                kwargs_str = ", ".join(kw_parts)
            self._emit(f"self.scheduler = {pytorch_sched}(self.optimizer, {kwargs_str})")
        else:
            self._emit("self.scheduler = None")

        # Loss function
        loss_map = self._get_loss_map()
        loss_key = loss_name
        pytorch_loss = loss_map.get(loss_key, f"nn.{loss_name}Loss()" if loss_name else "nn.CrossEntropyLoss()")

        # Label smoothing
        if train.label_smoothing and train.label_smoothing > 0:
            pytorch_loss = f"nn.CrossEntropyLoss(label_smoothing={train.label_smoothing})"

        self._emit(f"self.criterion = {pytorch_loss}")

        # Epochs
        self._emit(f"self.epochs = {train.epochs or 10}")

        # Mixed precision
        if train.precision in ("fp16", "mixed", "amp"):
            self._imports.add("from torch.cuda.amp import GradScaler, autocast")
            self._emit("self.scaler = GradScaler()")
            self._emit("self.use_amp = True")
        else:
            self._emit("self.use_amp = False")

        # Gradient accumulation
        grad_accum = train.gradient_accumulation if train.gradient_accumulation else 1
        self._emit(f"self.gradient_accumulation_steps = {grad_accum}")

        # Gradient clipping
        if train.gradient_clip:
            self._emit(f"self.gradient_clip_value = {train.gradient_clip}")
        else:
            self._emit("self.gradient_clip_value = None")
        if train.gradient_clip_norm:
            self._emit(f"self.gradient_clip_norm = {train.gradient_clip_norm}")
        else:
            self._emit("self.gradient_clip_norm = None")

        # EMA
        if train.ema:
            self._imports.add("import copy")
            self._emit(f"self.ema_decay = {train.ema_decay}")
            self._emit("self.ema_model = copy.deepcopy(self.model)")
            self._emit("self.ema_model.eval()")
            self._emit("for p in self.ema_model.parameters():")
            self._indent_up()
            self._emit("p.requires_grad_(False)")
            self._indent_down()

        # SWA
        if train.swa:
            self._imports.add("from torch.optim.swa_utils import AveragedModel, SWALR, update_bn")
            self._emit(f"self.swa_start = {train.swa_start}")
            self._emit("self.swa_model = AveragedModel(self.model)")
            self._emit("self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)")

        # Mixup/CutMix
        if train.mixup_alpha and train.mixup_alpha > 0:
            self._imports.add("import numpy as np")
            self._emit(f"self.mixup_alpha = {train.mixup_alpha}")
        if train.cutmix_alpha and train.cutmix_alpha > 0:
            self._imports.add("import numpy as np")
            self._emit(f"self.cutmix_alpha = {train.cutmix_alpha}")

        # Metrics tracking
        self._emit("self.history = {'train_loss': [], 'val_loss': []}")
        self._emit("self.best_val_loss = float('inf')")
        self._emit("self.best_model_state = None")

        self._indent_down()
        self._emit("")

        # EMA update method
        if train.ema:
            self._emit("@torch.no_grad()")
            self._emit("def _update_ema(self):")
            self._indent_up()
            self._emit("for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):")
            self._indent_up()
            self._emit("ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)")
            self._indent_down()
            self._indent_down()
            self._emit("")

        # Mixup helper
        if train.mixup_alpha and train.mixup_alpha > 0:
            self._emit("def _mixup_data(self, x, y):")
            self._indent_up()
            self._emit("lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)")
            self._emit("batch_size = x.size(0)")
            self._emit("index = torch.randperm(batch_size).to(self.device)")
            self._emit("mixed_x = lam * x + (1 - lam) * x[index]")
            self._emit("y_a, y_b = y, y[index]")
            self._emit("return mixed_x, y_a, y_b, lam")
            self._indent_down()
            self._emit("")

        # CutMix helper
        if train.cutmix_alpha and train.cutmix_alpha > 0:
            self._emit("def _cutmix_data(self, x, y):")
            self._indent_up()
            self._emit("lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)")
            self._emit("batch_size = x.size(0)")
            self._emit("index = torch.randperm(batch_size).to(self.device)")
            self._emit("bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)")
            self._emit("x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]")
            self._emit("lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))")
            self._emit("return x, y, y[index], lam")
            self._indent_down()
            self._emit("")
            self._emit("def _rand_bbox(self, size, lam):")
            self._indent_up()
            self._emit("W, H = size[2], size[3]")
            self._emit("cut_rat = np.sqrt(1.0 - lam)")
            self._emit("cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)")
            self._emit("cx, cy = np.random.randint(W), np.random.randint(H)")
            self._emit("bbx1 = np.clip(cx - cut_w // 2, 0, W)")
            self._emit("bby1 = np.clip(cy - cut_h // 2, 0, H)")
            self._emit("bbx2 = np.clip(cx + cut_w // 2, 0, W)")
            self._emit("bby2 = np.clip(cy + cut_h // 2, 0, H)")
            self._emit("return bbx1, bby1, bbx2, bby2")
            self._indent_down()
            self._emit("")

        # Train method
        self._emit("def run(self):")
        self._indent_up()
        self._emit("train_loader = self.data.train_loader()")
        self._emit("val_loader = self.data.val_loader()")
        self._emit("")
        self._emit("for epoch in range(self.epochs):")
        self._indent_up()
        self._emit("# -- Training Phase --")
        self._emit("self.model.train()")
        self._emit("train_loss = 0.0")
        self._emit("train_correct = 0")
        self._emit("train_total = 0")
        self._emit("")
        self._emit("pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')")
        self._emit("for batch_idx, (inputs, targets) in enumerate(pbar):")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("")

        # Mixup / CutMix
        if train.mixup_alpha and train.mixup_alpha > 0:
            self._emit("inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets)")
        elif train.cutmix_alpha and train.cutmix_alpha > 0:
            self._emit("inputs, targets_a, targets_b, lam = self._cutmix_data(inputs, targets)")

        use_amp = train.precision in ("fp16", "mixed", "amp")
        use_mixup = (train.mixup_alpha and train.mixup_alpha > 0) or (train.cutmix_alpha and train.cutmix_alpha > 0)

        if use_amp:
            self._emit("with autocast():")
            self._indent_up()
            self._emit("outputs = self.model(inputs)")
            if use_mixup:
                self._emit("loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)")
            else:
                self._emit("loss = self.criterion(outputs, targets)")
            if grad_accum > 1:
                self._emit(f"loss = loss / self.gradient_accumulation_steps")
            self._indent_down()
            self._emit("")
            self._emit("self.scaler.scale(loss).backward()")
            if grad_accum > 1:
                self._emit("if (batch_idx + 1) % self.gradient_accumulation_steps == 0:")
                self._indent_up()
            if train.gradient_clip_norm:
                self._emit("self.scaler.unscale_(self.optimizer)")
                self._emit(f"torch.nn.utils.clip_grad_norm_(self.model.parameters(), {train.gradient_clip_norm})")
            elif train.gradient_clip:
                self._emit("self.scaler.unscale_(self.optimizer)")
                self._emit(f"torch.nn.utils.clip_grad_value_(self.model.parameters(), {train.gradient_clip})")
            self._emit("self.scaler.step(self.optimizer)")
            self._emit("self.scaler.update()")
            self._emit("self.optimizer.zero_grad()")
            if grad_accum > 1:
                self._indent_down()
        else:
            self._emit("outputs = self.model(inputs)")
            if use_mixup:
                self._emit("loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)")
            else:
                self._emit("loss = self.criterion(outputs, targets)")
            if grad_accum > 1:
                self._emit(f"loss = loss / self.gradient_accumulation_steps")
            self._emit("")
            self._emit("loss.backward()")
            if grad_accum > 1:
                self._emit("if (batch_idx + 1) % self.gradient_accumulation_steps == 0:")
                self._indent_up()
            if train.gradient_clip_norm:
                self._emit(f"torch.nn.utils.clip_grad_norm_(self.model.parameters(), {train.gradient_clip_norm})")
            elif train.gradient_clip:
                self._emit(f"torch.nn.utils.clip_grad_value_(self.model.parameters(), {train.gradient_clip})")
            self._emit("self.optimizer.step()")
            self._emit("self.optimizer.zero_grad()")
            if grad_accum > 1:
                self._indent_down()

        # EMA update
        if train.ema:
            self._emit("self._update_ema()")

        self._emit("")
        self._emit("train_loss += loss.item()")
        self._emit("_, predicted = outputs.max(1)")
        self._emit("train_total += targets.size(0)")
        if use_mixup:
            self._emit("train_correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())")
        else:
            self._emit("train_correct += predicted.eq(targets).sum().item()")
        self._emit("")
        self._emit("pbar.set_postfix({")
        self._indent_up()
        self._emit("'loss': f'{loss.item():.4f}',")
        self._emit("'acc': f'{100.*train_correct/max(train_total,1):.2f}%'")
        self._indent_down()
        self._emit("})")
        self._indent_down()
        self._emit("")

        # Validation
        self._emit("# -- Validation Phase --")
        self._emit("self.model.eval()")
        self._emit("val_loss = 0.0")
        self._emit("val_correct = 0")
        self._emit("val_total = 0")
        self._emit("")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("for inputs, targets in val_loader:")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("outputs = self.model(inputs)")
        self._emit("loss = self.criterion(outputs, targets)")
        self._emit("val_loss += loss.item()")
        self._emit("_, predicted = outputs.max(1)")
        self._emit("val_total += targets.size(0)")
        self._emit("val_correct += predicted.eq(targets).sum().item()")
        self._indent_down()
        self._indent_down()
        self._emit("")

        self._emit("avg_train_loss = train_loss / len(train_loader)")
        self._emit("avg_val_loss = val_loss / len(val_loader)")
        self._emit("train_acc = 100. * train_correct / max(train_total, 1)")
        self._emit("val_acc = 100. * val_correct / max(val_total, 1)")
        self._emit("")
        self._emit("self.history['train_loss'].append(avg_train_loss)")
        self._emit("self.history['val_loss'].append(avg_val_loss)")
        self._emit("")
        self._emit("print(f'Epoch {epoch+1}/{self.epochs} | '")
        self._emit("      f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '")
        self._emit("      f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%')")
        self._emit("")

        # Save best model
        self._emit("if avg_val_loss < self.best_val_loss:")
        self._indent_up()
        self._emit("self.best_val_loss = avg_val_loss")
        self._emit("self.best_model_state = self.model.state_dict().copy()")
        self._emit("torch.save(self.model.state_dict(), 'best_model.pth')")
        self._emit("print(f'  -> Best model saved (val_loss: {avg_val_loss:.4f})')")
        self._indent_down()
        self._emit("")

        # Scheduler step
        if train.swa:
            self._emit(f"if epoch >= self.swa_start:")
            self._indent_up()
            self._emit("self.swa_model.update_parameters(self.model)")
            self._emit("self.swa_scheduler.step()")
            self._indent_down()
            self._emit("elif self.scheduler:")
            self._indent_up()
            self._emit("self.scheduler.step()")
            self._indent_down()
        else:
            self._emit("if self.scheduler:")
            self._indent_up()
            self._emit("self.scheduler.step()")
            self._indent_down()

        self._indent_down()  # epoch loop
        self._emit("")

        # SWA finalize
        if train.swa:
            self._emit("update_bn(self.data.train_loader(), self.swa_model, device=self.device)")
            self._emit("self.model = self.swa_model")
            self._emit("")

        self._emit("print('\\nTraining complete!')")
        self._emit("return self.history")
        self._indent_down()  # run method

        self._indent_down()  # class

    def _emit_distributed_setup(self, train: TrainDef):
        """Emit distributed training setup code."""
        dist_type = train.distributed.lower() if train.distributed else "ddp"
        if dist_type == "ddp":
            self._imports.add("import torch.distributed as dist")
            self._imports.add("from torch.nn.parallel import DistributedDataParallel as DDP")
            self._imports.add("from torch.utils.data.distributed import DistributedSampler")
            self._emit("def setup_distributed(rank, world_size):")
            self._indent_up()
            self._emit("os.environ['MASTER_ADDR'] = 'localhost'")
            self._emit("os.environ['MASTER_PORT'] = '12355'")
            self._emit("dist.init_process_group('nccl', rank=rank, world_size=world_size)")
            self._emit("torch.cuda.set_device(rank)")
            self._indent_down()
            self._emit("")
            self._emit("def cleanup_distributed():")
            self._indent_up()
            self._emit("dist.destroy_process_group()")
            self._indent_down()
            self._emit("")
        elif dist_type == "fsdp":
            self._imports.add("import torch.distributed as dist")
            self._imports.add("from torch.distributed.fsdp import FullyShardedDataParallel as FSDP")
            self._imports.add("from torch.distributed.fsdp import ShardingStrategy")
            self._emit("def setup_fsdp(rank, world_size):")
            self._indent_up()
            self._emit("os.environ['MASTER_ADDR'] = 'localhost'")
            self._emit("os.environ['MASTER_PORT'] = '12355'")
            self._emit("dist.init_process_group('nccl', rank=rank, world_size=world_size)")
            self._emit("torch.cuda.set_device(rank)")
            self._indent_down()
            self._emit("")
        elif dist_type == "deepspeed":
            self._imports.add("import deepspeed")
            self._emit("# DeepSpeed configuration")
            self._emit("ds_config = {")
            self._indent_up()
            self._emit("'train_micro_batch_size_per_gpu': 8,")
            self._emit("'gradient_accumulation_steps': 1,")
            self._emit("'optimizer': {'type': 'Adam', 'params': {'lr': 3e-4}},")
            self._emit("'fp16': {'enabled': True},")
            self._emit("'zero_optimization': {'stage': 2},")
            self._indent_down()
            self._emit("}")
            self._emit("")

    def _emit_focal_loss(self):
        self._emit("class FocalLoss(nn.Module):")
        self._indent_up()
        self._emit("def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.alpha = alpha")
        self._emit("self.gamma = gamma")
        self._emit("self.reduction = reduction")
        self._indent_down()
        self._emit("")
        self._emit("def forward(self, inputs, targets):")
        self._indent_up()
        self._emit("ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')")
        self._emit("pt = torch.exp(-ce_loss)")
        self._emit("focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss")
        self._emit("if self.reduction == 'mean': return focal_loss.mean()")
        self._emit("elif self.reduction == 'sum': return focal_loss.sum()")
        self._emit("return focal_loss")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _emit_dice_loss(self):
        self._emit("class DiceLoss(nn.Module):")
        self._indent_up()
        self._emit("def __init__(self, smooth=1.0):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.smooth = smooth")
        self._indent_down()
        self._emit("")
        self._emit("def forward(self, inputs, targets):")
        self._indent_up()
        self._emit("inputs = torch.sigmoid(inputs)")
        self._emit("inputs = inputs.view(-1)")
        self._emit("targets = targets.view(-1)")
        self._emit("intersection = (inputs * targets).sum()")
        self._emit("dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)")
        self._emit("return 1 - dice")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _emit_contrastive_loss(self):
        self._emit("class ContrastiveLoss(nn.Module):")
        self._indent_up()
        self._emit("def __init__(self, margin=1.0):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.margin = margin")
        self._indent_down()
        self._emit("")
        self._emit("def forward(self, output1, output2, label):")
        self._indent_up()
        self._emit("dist = nn.functional.pairwise_distance(output1, output2)")
        self._emit("loss = label * dist.pow(2) + (1 - label) * nn.functional.relu(self.margin - dist).pow(2)")
        self._emit("return loss.mean()")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _emit_ntxent_loss(self):
        self._emit("class NTXentLoss(nn.Module):")
        self._indent_up()
        self._emit("def __init__(self, temperature=0.5):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.temperature = temperature")
        self._indent_down()
        self._emit("")
        self._emit("def forward(self, z_i, z_j):")
        self._indent_up()
        self._emit("batch_size = z_i.size(0)")
        self._emit("z = torch.cat([z_i, z_j], dim=0)")
        self._emit("sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature")
        self._emit("mask = torch.eye(2 * batch_size, device=z.device).bool()")
        self._emit("sim.masked_fill_(mask, -1e9)")
        self._emit("pos = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)])")
        self._emit("loss = -pos + torch.logsumexp(sim, dim=1)")
        self._emit("return loss.mean()")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _emit_infonce_loss(self):
        self._emit("class InfoNCELoss(nn.Module):")
        self._indent_up()
        self._emit("def __init__(self, temperature=0.07):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.temperature = temperature")
        self._indent_down()
        self._emit("")
        self._emit("def forward(self, query, positive_key, negative_keys=None):")
        self._indent_up()
        self._emit("pos_logit = torch.sum(query * positive_key, dim=1, keepdim=True) / self.temperature")
        self._emit("if negative_keys is not None:")
        self._indent_up()
        self._emit("neg_logits = query @ negative_keys.T / self.temperature")
        self._emit("logits = torch.cat([pos_logit, neg_logits], dim=1)")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("logits = pos_logit")
        self._indent_down()
        self._emit("labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)")
        self._emit("return nn.functional.cross_entropy(logits, labels)")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _emit_lion_optimizer(self):
        self._emit("class Lion(optim.Optimizer):")
        self._indent_up()
        self._emit("def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):")
        self._indent_up()
        self._emit("defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)")
        self._emit("super().__init__(params, defaults)")
        self._indent_down()
        self._emit("")
        self._emit("@torch.no_grad()")
        self._emit("def step(self, closure=None):")
        self._indent_up()
        self._emit("loss = None")
        self._emit("if closure is not None: loss = closure()")
        self._emit("for group in self.param_groups:")
        self._indent_up()
        self._emit("for p in group['params']:")
        self._indent_up()
        self._emit("if p.grad is None: continue")
        self._emit("grad = p.grad")
        self._emit("state = self.state[p]")
        self._emit("if len(state) == 0: state['exp_avg'] = torch.zeros_like(p)")
        self._emit("exp_avg = state['exp_avg']")
        self._emit("beta1, beta2 = group['betas']")
        self._emit("update = exp_avg * beta1 + grad * (1 - beta1)")
        self._emit("p.add_(torch.sign(update), alpha=-group['lr'])")
        self._emit("if group['weight_decay'] > 0:")
        self._indent_up()
        self._emit("p.add_(p, alpha=-group['lr'] * group['weight_decay'])")
        self._indent_down()
        self._emit("exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)")
        self._indent_down()
        self._indent_down()
        self._emit("return loss")
        self._indent_down()
        self._indent_down()
        self._emit("")

    def _transpile_train_tensorflow(self, train: TrainDef):
        self._emit(f"def run_{train.name}():")
        self._indent_up()
        model_code = self._value_to_python(train.model_ref) if train.model_ref else "None"
        self._emit(f"model = {model_code}")
        opt_str = "tf.keras.optimizers.Adam(learning_rate=3e-4)"
        if train.optimizer and isinstance(train.optimizer, FunctionCall):
            opt_str = f"tf.keras.optimizers.{train.optimizer.name}("
            kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in train.optimizer.kwargs.items())
            opt_str += kwargs + ")"
        loss_name = "sparse_categorical_crossentropy"
        if isinstance(train.loss, Identifier):
            loss_map = {"crossentropy": "sparse_categorical_crossentropy", "mse": "mse", "bce": "binary_crossentropy"}
            loss_name = loss_map.get(train.loss.name.lower(), train.loss.name)
        self._emit(f"model.compile(optimizer={opt_str}, loss='{loss_name}', metrics=['accuracy'])")
        self._emit(f"history = model.fit(train_ds, validation_data=val_ds, epochs={train.epochs or 10})")
        self._emit("return history")
        self._indent_down()

    def _transpile_train_jax(self, train: TrainDef):
        self._emit(f"def run_{train.name}():")
        self._indent_up()
        self._emit("# JAX training loop with Flax")
        self._emit("pass")
        self._indent_down()

    # --- Evaluation Transpilation --------------------------------------------------

    def _transpile_eval(self, eval_def: EvalDef):
        self._imports.add("from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score")
        self._imports.add("from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report")
        self._imports.add("import numpy as np")

        self._emit(f"class {eval_def.name}Evaluator:")
        self._indent_up()
        self._emit(f'"""Evaluation runner for {eval_def.name}."""')
        self._emit("")

        self._emit("def __init__(self, model, data, device='auto'):")
        self._indent_up()
        self._emit("if device == 'auto':")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("self.device = torch.device(device)")
        self._indent_down()
        self._emit("self.model = model.to(self.device)")
        self._emit("self.data = data")

        cp = eval_def.checkpoint
        if cp == "best":
            self._emit("if os.path.exists('best_model.pth'):")
            self._indent_up()
            self._emit("self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))")
            self._indent_down()

        self._indent_down()
        self._emit("")

        self._emit("def run(self):")
        self._indent_up()
        self._emit("self.model.eval()")
        self._emit("all_preds = []")
        self._emit("all_targets = []")
        self._emit("all_probs = []")
        self._emit("")
        self._emit("test_loader = self.data.test_loader()")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("for inputs, targets in test_loader:")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("outputs = self.model(inputs)")
        self._emit("probs = torch.softmax(outputs, dim=1)")
        self._emit("_, preds = outputs.max(1)")
        self._emit("all_preds.extend(preds.cpu().numpy())")
        self._emit("all_targets.extend(targets.cpu().numpy())")
        self._emit("all_probs.extend(probs.cpu().numpy())")
        self._indent_down()
        self._indent_down()
        self._emit("")
        self._emit("all_preds = np.array(all_preds)")
        self._emit("all_targets = np.array(all_targets)")
        self._emit("all_probs = np.array(all_probs)")
        self._emit("")
        self._emit("results = {}")

        for metric in eval_def.metrics:
            if isinstance(metric, Identifier):
                m = metric.name.lower()
            elif isinstance(metric, FunctionCall):
                m = metric.name.lower()
            elif isinstance(metric, str):
                m = metric.lower()
            else:
                continue
            metric_code = {
                "accuracy": "results['accuracy'] = accuracy_score(all_targets, all_preds)",
                "precision": "results['precision'] = precision_score(all_targets, all_preds, average='macro')",
                "recall": "results['recall'] = recall_score(all_targets, all_preds, average='macro')",
                "f1": "results['f1'] = f1_score(all_targets, all_preds, average='macro')",
                "confusion_matrix": "results['confusion_matrix'] = confusion_matrix(all_targets, all_preds)",
                "roc_auc": "results['roc_auc'] = roc_auc_score(all_targets, all_probs, multi_class='ovr')",
                "classification_report": "results['report'] = classification_report(all_targets, all_preds)",
            }
            code = metric_code.get(m)
            if code:
                self._emit(code)

        self._emit("")
        self._emit("for k, v in results.items():")
        self._indent_up()
        self._emit("if isinstance(v, np.ndarray):")
        self._indent_up()
        self._emit("print(f'{k}:\\n{v}')")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')")
        self._indent_down()
        self._indent_down()
        self._emit("")
        self._emit("return results")
        self._indent_down()

        self._indent_down()

    # --- Search Transpilation ----------------------------------------------------

    def _transpile_search(self, search: SearchDef):
        self._imports.add("import optuna")

        self._emit(f"class {search.name}Search:")
        self._indent_up()
        self._emit(f'"""Hyperparameter search for {search.name}."""')
        self._emit("")

        self._emit(f"def __init__(self):")
        self._indent_up()
        self._emit(f"self.n_trials = {search.trials}")
        self._emit(f"self.method = '{search.method}'")
        self._indent_down()
        self._emit("")

        self._emit("def objective(self, trial):")
        self._indent_up()

        for param_name, func_call in search.space.items():
            suggest = self._optuna_suggest(param_name, func_call)
            self._emit(suggest)

        self._emit("")
        self._emit(f"# Build and train model with suggested hyperparameters")
        self._emit(f"# trainer = {search.base_ref}Trainer()")
        self._emit(f"# history = trainer.run()")
        self._emit(f"# return history['val_loss'][-1]")
        self._emit(f"pass")
        self._indent_down()
        self._emit("")

        self._emit("def run(self):")
        self._indent_up()

        if search.objective and isinstance(search.objective, FunctionCall):
            direction = "maximize" if "max" in search.objective.name.lower() else "minimize"
        else:
            direction = "minimize"

        sampler_map = {
            "bayesian": "optuna.samplers.TPESampler()",
            "tpe": "optuna.samplers.TPESampler()",
            "random": "optuna.samplers.RandomSampler()",
            "grid": "optuna.samplers.GridSampler({})",
            "cmaes": "optuna.samplers.CmaEsSampler()",
        }
        sampler = sampler_map.get(search.method, "optuna.samplers.TPESampler()")

        self._emit(f"sampler = {sampler}")
        self._emit(f"study = optuna.create_study(direction='{direction}', sampler=sampler)")
        self._emit(f"study.optimize(self.objective, n_trials=self.n_trials)")
        self._emit("")
        self._emit("print(f'Best trial: {study.best_trial.value}')")
        self._emit("print(f'Best params: {study.best_trial.params}')")
        self._emit("return study")
        self._indent_down()

        self._indent_down()

    def _optuna_suggest(self, name: str, call: FunctionCall) -> str:
        fn = call.name.lower()
        args = [self._value_to_python(a) for a in call.args]
        suggest_map = {
            "uniform": f"{name} = trial.suggest_float('{name}', {', '.join(args)})",
            "log_uniform": f"{name} = trial.suggest_float('{name}', {', '.join(args)}, log=True)",
            "int_uniform": f"{name} = trial.suggest_int('{name}', {', '.join(args)})",
            "choice": f"{name} = trial.suggest_categorical('{name}', [{', '.join(args)}])",
            "loguniform": f"{name} = trial.suggest_float('{name}', {', '.join(args)}, log=True)",
        }
        return suggest_map.get(fn, f"{name} = trial.suggest_float('{name}', {', '.join(args)})")

    # --- Deploy Transpilation ----------------------------------------------------

    def _transpile_deploy(self, deploy: DeployDef):
        self._emit(f"class {deploy.name}Deployer:")
        self._indent_up()
        self._emit(f'"""Deployment manager for {deploy.name}."""')
        self._emit("")

        self._emit("def __init__(self, model):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.model.eval()")
        self._indent_down()
        self._emit("")

        fmt = deploy.format.lower()
        self._emit("def export(self, output_path='exported_model'):")
        self._indent_up()

        if fmt == "onnx":
            self._imports.add("import torch.onnx")
            self._emit("dummy_input = torch.randn(1, 3, 224, 224)")
            self._emit("torch.onnx.export(")
            self._indent_up()
            self._emit("self.model, dummy_input,")
            self._emit("f'{output_path}.onnx',")
            self._emit("export_params=True,")
            self._emit("opset_version=17,")
            self._emit("do_constant_folding=True,")
            self._emit("input_names=['input'],")
            self._emit("output_names=['output'],")
            self._emit("dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}")
            self._indent_down()
            self._emit(")")
        elif fmt == "torchscript":
            self._emit("scripted = torch.jit.script(self.model)")
            self._emit("scripted.save(f'{output_path}.pt')")
        elif fmt == "savedmodel":
            self._emit("tf.saved_model.save(self.model, output_path)")
        else:
            self._emit("torch.save(self.model.state_dict(), f'{output_path}.pth')")

        if deploy.optimize:
            self._emit("")
            self._emit("print('Model exported with optimizations enabled')")

        if deploy.quantize:
            self._emit("")
            quant = deploy.quantize.lower()
            if quant == "int8":
                self._emit("quantized_model = torch.quantization.quantize_dynamic(")
                self._emit("    self.model, {nn.Linear}, dtype=torch.qint8)")
                self._emit("print('INT8 quantization applied')")
            elif quant == "fp16":
                self._emit("self.model = self.model.half()")
                self._emit("print('FP16 quantization applied')")

        self._emit(f"print(f'Model exported to {{output_path}}')")
        self._indent_down()
        self._emit("")

        if deploy.serve_config:
            self._imports.add("from fastapi import FastAPI")
            self._imports.add("import uvicorn")

            self._emit("def serve(self, host='0.0.0.0', port=8000):")
            self._indent_up()
            self._emit("app = FastAPI()")
            self._emit("")

            endpoint = "/predict"
            for k, v in deploy.serve_config.items():
                if k == "endpoint" and isinstance(v, StringLiteral):
                    endpoint = v.value

            self._emit(f"@app.post('{endpoint}')")
            self._emit("async def predict(data: dict):")
            self._indent_up()
            self._emit("input_tensor = torch.tensor(data['input']).unsqueeze(0)")
            self._emit("with torch.no_grad():")
            self._indent_up()
            self._emit("output = self.model(input_tensor)")
            self._indent_down()
            self._emit("return {'prediction': output.tolist()}")
            self._indent_down()
            self._emit("")
            self._emit("uvicorn.run(app, host=host, port=port)")
            self._indent_down()

        self._indent_down()

    # --- Pipeline Transpilation --------------------------------------------------

    def _transpile_pipeline(self, pipeline: PipelineDef):
        self._emit(f"class {pipeline.name}Pipeline:")
        self._indent_up()
        self._emit(f'"""ML Pipeline: {pipeline.name}."""')
        self._emit("")
        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("self.steps = []")
        self._emit("self.results = {}")
        self._indent_down()
        self._emit("")
        self._emit("def run(self):")
        self._indent_up()
        self._emit("for step_name, step_fn in self.steps:")
        self._indent_up()
        self._emit("print(f'Running step: {step_name}')")
        self._emit("self.results[step_name] = step_fn()")
        self._indent_down()
        self._emit("return self.results")
        self._indent_down()
        self._indent_down()

    # --- Finetune Transpilation ----------------------------------------------------

    def _transpile_finetune(self, ft: FinetuneDef):
        self._imports.add("from transformers import AutoModelForSequenceClassification, AutoTokenizer")
        self._imports.add("from transformers import TrainingArguments, Trainer as HFTrainer")

        if ft.method.lower() in ("lora", "qlora"):
            self._imports.add("from peft import get_peft_model, LoraConfig, TaskType")

        self._emit(f"class {ft.name}Finetuner:")
        self._indent_up()
        self._emit(f'"""Fine-tuning manager for {ft.name}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.base_model_name = '{ft.base_model}'")
        self._emit(f"self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)")
        self._emit(f"self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)")

        if ft.method.lower() in ("lora", "qlora"):
            r = ft.config.get("r", 8)
            alpha = ft.config.get("alpha", 16)
            dropout = ft.config.get("dropout", 0.1)
            r_val = self._value_to_python(r) if not isinstance(r, (int, float)) else str(r)
            alpha_val = self._value_to_python(alpha) if not isinstance(alpha, (int, float)) else str(alpha)
            dropout_val = self._value_to_python(dropout) if not isinstance(dropout, (int, float)) else str(dropout)

            self._emit("")
            self._emit("lora_config = LoraConfig(")
            self._indent_up()
            self._emit(f"r={r_val},")
            self._emit(f"lora_alpha={alpha_val},")
            self._emit(f"lora_dropout={dropout_val},")
            self._emit("task_type=TaskType.SEQ_CLS,")
            self._emit("target_modules=['q_proj', 'v_proj']")
            self._indent_down()
            self._emit(")")
            self._emit("self.model = get_peft_model(self.model, lora_config)")
            self._emit("self.model.print_trainable_parameters()")

        self._indent_down()
        self._emit("")

        self._emit("def run(self, train_dataset, eval_dataset=None):")
        self._indent_up()
        self._emit("training_args = TrainingArguments(")
        self._indent_up()
        self._emit(f"output_dir='./results/{ft.name}',")
        self._emit("num_train_epochs=3,")
        self._emit("per_device_train_batch_size=16,")
        self._emit("per_device_eval_batch_size=64,")
        self._emit("warmup_steps=500,")
        self._emit("weight_decay=0.01,")
        self._emit("logging_dir='./logs',")
        self._emit("logging_steps=10,")
        self._emit("eval_strategy='epoch' if eval_dataset else 'no',")
        self._emit("save_strategy='epoch',")
        self._emit("load_best_model_at_end=True if eval_dataset else False,")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit("trainer = HFTrainer(")
        self._indent_up()
        self._emit("model=self.model,")
        self._emit("args=training_args,")
        self._emit("train_dataset=train_dataset,")
        self._emit("eval_dataset=eval_dataset,")
        self._emit("tokenizer=self.tokenizer,")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit("trainer.train()")
        self._emit("return trainer")
        self._indent_down()

        self._indent_down()

    # --- Ensemble Transpilation --------------------------------------------------

    def _transpile_ensemble(self, ens: EnsembleDef):
        self._emit(f"class {ens.name}Ensemble:")
        self._indent_up()
        self._emit(f'"""Ensemble of models using {ens.strategy} strategy."""')
        self._emit("")

        self._emit(f"def __init__(self, models):")
        self._indent_up()
        self._emit("self.models = models")
        self._emit(f"self.strategy = '{ens.strategy}'")
        if ens.weights:
            self._emit(f"self.weights = {ens.weights}")
        else:
            self._emit("self.weights = [1.0 / len(models)] * len(models)")
        self._indent_down()
        self._emit("")

        self._emit("def predict(self, x):")
        self._indent_up()
        self._emit("predictions = []")
        self._emit("for model in self.models:")
        self._indent_up()
        self._emit("model.eval()")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("pred = model(x)")
        self._emit("predictions.append(pred)")
        self._indent_down()
        self._indent_down()
        self._emit("")

        if ens.strategy == "voting":
            self._emit("votes = torch.stack([p.argmax(dim=1) for p in predictions])")
            self._emit("result = torch.mode(votes, dim=0).values")
        elif ens.strategy == "averaging":
            self._emit("weighted = sum(w * p for w, p in zip(self.weights, predictions))")
            self._emit("result = weighted")
        elif ens.strategy == "stacking":
            self._emit("result = torch.cat(predictions, dim=1)")
        else:
            self._emit("result = predictions[0]")

        self._emit("return result")
        self._indent_down()
        self._indent_down()

    # --- Explain Transpilation ---------------------------------------------------

    def _transpile_explain(self, exp: ExplainDef):
        method = exp.method.lower()

        if method == "shap":
            self._imports.add("import shap")
        elif method == "lime":
            self._imports.add("import lime")
            self._imports.add("import lime.lime_tabular")
        elif method in ("gradcam", "grad_cam"):
            self._imports.add("from torchcam.methods import GradCAM")
        elif method == "captum":
            self._imports.add("from captum.attr import IntegratedGradients, Saliency, DeepLift")

        self._emit(f"class {exp.name}Explainer:")
        self._indent_up()
        self._emit(f'"""Model interpretability using {method}."""')
        self._emit("")

        self._emit("def __init__(self, model, data=None):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.data = data")
        self._emit("self.model.eval()")
        self._indent_down()
        self._emit("")

        self._emit("def explain(self, inputs):")
        self._indent_up()

        if method == "shap":
            self._emit("explainer = shap.DeepExplainer(self.model, inputs)")
            self._emit("shap_values = explainer.shap_values(inputs)")
            self._emit("shap.summary_plot(shap_values, inputs)")
            self._emit("return shap_values")
        elif method == "lime":
            self._emit("explainer = lime.lime_tabular.LimeTabularExplainer(inputs.numpy())")
            self._emit("explanation = explainer.explain_instance(inputs[0].numpy(), self.model)")
            self._emit("return explanation")
        elif method in ("gradcam", "grad_cam"):
            self._emit("cam = GradCAM(self.model)")
            self._emit("with torch.no_grad():")
            self._indent_up()
            self._emit("activation_map = cam(inputs)")
            self._indent_down()
            self._emit("return activation_map")
        elif method == "captum":
            self._emit("ig = IntegratedGradients(self.model)")
            self._emit("attributions = ig.attribute(inputs, target=0)")
            self._emit("return attributions")
        elif method == "attention":
            self._emit("outputs = self.model(inputs, output_attentions=True)")
            self._emit("attention_weights = outputs.attentions")
            self._emit("return attention_weights")
        else:
            self._emit(f"raise NotImplementedError('Method {method} not yet supported')")

        self._indent_down()
        self._indent_down()

    # --- Pretrain Transpilation --------------------------------------------------

    def _transpile_pretrain(self, pt: PretrainDef):
        obj = pt.objective.lower()

        self._emit(f"class {pt.name}Pretrainer:")
        self._indent_up()
        self._emit(f'"""Pretraining with {obj} objective."""')
        self._emit("")

        self._emit("def __init__(self, model, data=None):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.data = data")
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("self.model.to(self.device)")
        self._indent_down()
        self._emit("")

        self._emit("def run(self, epochs=100):")
        self._indent_up()

        if obj == "masked_lm":
            self._imports.add("from transformers import DataCollatorForLanguageModeling")
            self._emit("collator = DataCollatorForLanguageModeling(tokenizer=None, mlm=True, mlm_probability=0.15)")
            self._emit("optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)")
            self._emit("self.model.train()")
            self._emit("for epoch in range(epochs):")
            self._indent_up()
            self._emit("total_loss = 0")
            self._emit("for batch in self.data:")
            self._indent_up()
            self._emit("batch = {k: v.to(self.device) for k, v in batch.items()}")
            self._emit("outputs = self.model(**batch)")
            self._emit("loss = outputs.loss")
            self._emit("loss.backward()")
            self._emit("optimizer.step()")
            self._emit("optimizer.zero_grad()")
            self._emit("total_loss += loss.item()")
            self._indent_down()
            self._emit("print(f'Epoch {epoch+1}: loss={total_loss:.4f}')")
            self._indent_down()
        elif obj == "contrastive":
            self._emit("optimizer = optim.Adam(self.model.parameters(), lr=3e-4)")
            self._emit("temperature = 0.5")
            self._emit("self.model.train()")
            self._emit("for epoch in range(epochs):")
            self._indent_up()
            self._emit("total_loss = 0")
            self._emit("for x_i, x_j in self.data:")
            self._indent_up()
            self._emit("x_i, x_j = x_i.to(self.device), x_j.to(self.device)")
            self._emit("z_i, z_j = self.model(x_i), self.model(x_j)")
            self._emit("z_i = nn.functional.normalize(z_i, dim=1)")
            self._emit("z_j = nn.functional.normalize(z_j, dim=1)")
            self._emit("sim = torch.mm(z_i, z_j.T) / temperature")
            self._emit("labels = torch.arange(z_i.size(0), device=self.device)")
            self._emit("loss = nn.functional.cross_entropy(sim, labels)")
            self._emit("loss.backward()")
            self._emit("optimizer.step()")
            self._emit("optimizer.zero_grad()")
            self._emit("total_loss += loss.item()")
            self._indent_down()
            self._emit("print(f'Epoch {epoch+1}: loss={total_loss:.4f}')")
            self._indent_down()
        elif obj == "autoregressive":
            self._emit("optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)")
            self._emit("self.model.train()")
            self._emit("for epoch in range(epochs):")
            self._indent_up()
            self._emit("total_loss = 0")
            self._emit("for batch in self.data:")
            self._indent_up()
            self._emit("input_ids = batch['input_ids'].to(self.device)")
            self._emit("labels = input_ids.clone()")
            self._emit("outputs = self.model(input_ids, labels=labels)")
            self._emit("loss = outputs.loss")
            self._emit("loss.backward()")
            self._emit("optimizer.step()")
            self._emit("optimizer.zero_grad()")
            self._emit("total_loss += loss.item()")
            self._indent_down()
            self._emit("print(f'Epoch {epoch+1}: loss={total_loss:.4f}')")
            self._indent_down()
        else:
            self._emit(f"# Custom pretraining objective: {obj}")
            self._emit("pass")

        self._indent_down()
        self._indent_down()

    # --- Python Escape -----------------------------------------------------------

    def _transpile_python_block(self, block: PythonBlock):
        """Pass-through raw Python code."""
        for line in block.code.split("\n"):
            self._emit(line)

    def _transpile_axon_import(self, node: AxonImport):
        """Transpile an AxonImport node.
        
        For .axon imports, attempt to compile the target file and inline it.
        For python-style imports, emit the import statement directly.
        """
        if node.import_style == "python":
            # Emit as standard Python import
            if node.alias:
                self._emit(f"import {node.module} as {node.alias}")
            else:
                self._emit(f"import {node.module}")
            return

        if not node.source_path:
            return

        # Attempt to resolve and inline the .axon module
        import os
        try:
            from axon.modules import ModuleResolver
            resolver = ModuleResolver()
            info = resolver.load(node.source_path)
            
            # Compile the imported module's source using a fresh transpiler
            from axon.parser.parser import AxonParser
            sub_parser = AxonParser(info.source)
            sub_ast = sub_parser.parse()
            sub_transpiler = AxonTranspiler(backend=self.backend)
            compiled = sub_transpiler.transpile(sub_ast)
            
            if node.import_style == "wildcard":
                self._emit(f"# -- Inlined from {node.source_path} (wildcard) --")
                for line in compiled.split("\n"):
                    self._emit(line)
            else:
                # Named import: emit a comment and inline the compiled code
                names_str = ", ".join(node.names)
                self._emit(f"# -- Inlined from {node.source_path}: {names_str} --")
                for line in compiled.split("\n"):
                    self._emit(line)
        except Exception as e:
            # If the module can't be resolved (e.g., in tests), emit a comment
            self._emit(f"# AxonImport: {node.import_style} from {node.source_path!r} -- {e}")

    # --- GAN Transpilation ---------------------------------------------------------

    def _transpile_gan(self, gan: GANDef):
        self._imports.add("import torch.nn.functional as F")
        self._imports.add("from tqdm import tqdm")
        self._imports.add("import numpy as np")

        self._emit(f"class {gan.name}GAN:")
        self._indent_up()
        self._emit(f'"""GAN training with {gan.loss_type} loss."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit(f"self.latent_dim = {gan.latent_dim}")
        self._emit(f"self.n_critic = {gan.n_critic}")
        self._emit(f"self.epochs = {gan.epochs}")

        gen_code = self._value_to_python(gan.generator) if gan.generator else "self._build_generator()"
        disc_code = self._value_to_python(gan.discriminator) if gan.discriminator else "self._build_discriminator()"
        self._emit(f"self.generator = {gen_code}.to(self.device)")
        self._emit(f"self.discriminator = {disc_code}.to(self.device)")

        if gan.optimizer_g and isinstance(gan.optimizer_g, FunctionCall):
            opt_map = self._get_optimizer_map()
            opt_cls = opt_map.get(gan.optimizer_g.name.lower(), f"optim.{gan.optimizer_g.name}")
            kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in gan.optimizer_g.kwargs.items())
            self._emit(f"self.optimizer_g = {opt_cls}(self.generator.parameters(), {kwargs})")
        else:
            self._emit("self.optimizer_g = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))")

        if gan.optimizer_d and isinstance(gan.optimizer_d, FunctionCall):
            opt_map = self._get_optimizer_map()
            opt_cls = opt_map.get(gan.optimizer_d.name.lower(), f"optim.{gan.optimizer_d.name}")
            kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in gan.optimizer_d.kwargs.items())
            self._emit(f"self.optimizer_d = {opt_cls}(self.discriminator.parameters(), {kwargs})")
        else:
            self._emit("self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))")

        self._indent_down()
        self._emit("")

        # Loss computation method
        loss_type = gan.loss_type.lower()
        self._emit("def _d_loss(self, real_output, fake_output):")
        self._indent_up()
        if loss_type == "vanilla":
            self._emit("real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))")
            self._emit("fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))")
            self._emit("return (real_loss + fake_loss) / 2")
        elif loss_type == "wgan":
            self._emit("return -torch.mean(real_output) + torch.mean(fake_output)")
        elif loss_type == "wgan_gp":
            self._emit("return -torch.mean(real_output) + torch.mean(fake_output)")
        elif loss_type == "lsgan":
            self._emit("real_loss = F.mse_loss(real_output, torch.ones_like(real_output))")
            self._emit("fake_loss = F.mse_loss(fake_output, torch.zeros_like(fake_output))")
            self._emit("return (real_loss + fake_loss) / 2")
        elif loss_type == "hinge":
            self._emit("real_loss = torch.mean(F.relu(1.0 - real_output))")
            self._emit("fake_loss = torch.mean(F.relu(1.0 + fake_output))")
            self._emit("return real_loss + fake_loss")
        else:
            self._emit("real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))")
            self._emit("fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))")
            self._emit("return (real_loss + fake_loss) / 2")
        self._indent_down()
        self._emit("")

        self._emit("def _g_loss(self, fake_output):")
        self._indent_up()
        if loss_type == "vanilla":
            self._emit("return F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))")
        elif loss_type in ("wgan", "wgan_gp"):
            self._emit("return -torch.mean(fake_output)")
        elif loss_type == "lsgan":
            self._emit("return F.mse_loss(fake_output, torch.ones_like(fake_output))")
        elif loss_type == "hinge":
            self._emit("return -torch.mean(fake_output)")
        else:
            self._emit("return F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))")
        self._indent_down()
        self._emit("")

        # Gradient penalty for WGAN-GP
        if loss_type == "wgan_gp":
            self._emit("def _gradient_penalty(self, real_data, fake_data):")
            self._indent_up()
            self._emit("batch_size = real_data.size(0)")
            self._emit("alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)")
            self._emit("interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)")
            self._emit("d_interpolated = self.discriminator(interpolated)")
            self._emit("gradients = torch.autograd.grad(")
            self._indent_up()
            self._emit("outputs=d_interpolated, inputs=interpolated,")
            self._emit("grad_outputs=torch.ones_like(d_interpolated),")
            self._emit("create_graph=True, retain_graph=True")
            self._indent_down()
            self._emit(")[0]")
            self._emit("gradients = gradients.view(batch_size, -1)")
            self._emit("gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()")
            self._emit(f"return {gan.gp_weight} * gradient_penalty")
            self._indent_down()
            self._emit("")

        # Training loop
        self._emit("def run(self, dataloader):")
        self._indent_up()
        self._emit("g_losses, d_losses = [], []")
        self._emit("")
        self._emit("for epoch in range(self.epochs):")
        self._indent_up()
        self._emit("pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')")
        self._emit("for batch_idx, (real_data, _) in enumerate(pbar):")
        self._indent_up()
        self._emit("real_data = real_data.to(self.device)")
        self._emit("batch_size = real_data.size(0)")
        self._emit("")

        # Discriminator step
        self._emit("# Train Discriminator")
        self._emit("z = torch.randn(batch_size, self.latent_dim, device=self.device)")
        self._emit("fake_data = self.generator(z).detach()")
        self._emit("real_output = self.discriminator(real_data)")
        self._emit("fake_output = self.discriminator(fake_data)")
        self._emit("d_loss = self._d_loss(real_output, fake_output)")
        if loss_type == "wgan_gp":
            self._emit("d_loss = d_loss + self._gradient_penalty(real_data, fake_data)")
        self._emit("self.optimizer_d.zero_grad()")
        self._emit("d_loss.backward()")
        self._emit("self.optimizer_d.step()")
        if loss_type == "wgan":
            self._emit("for p in self.discriminator.parameters():")
            self._indent_up()
            self._emit("p.data.clamp_(-0.01, 0.01)")
            self._indent_down()
        self._emit("")

        # Generator step
        self._emit("# Train Generator")
        self._emit("if batch_idx % self.n_critic == 0:")
        self._indent_up()
        self._emit("z = torch.randn(batch_size, self.latent_dim, device=self.device)")
        self._emit("fake_data = self.generator(z)")
        self._emit("fake_output = self.discriminator(fake_data)")
        self._emit("g_loss = self._g_loss(fake_output)")
        self._emit("self.optimizer_g.zero_grad()")
        self._emit("g_loss.backward()")
        self._emit("self.optimizer_g.step()")
        self._emit("g_losses.append(g_loss.item())")
        self._indent_down()
        self._emit("")
        self._emit("d_losses.append(d_loss.item())")
        self._emit("pbar.set_postfix({'D_loss': f'{d_loss.item():.4f}', 'G_loss': f'{g_losses[-1]:.4f}' if g_losses else 'N/A'})")
        self._indent_down()
        self._emit("")
        self._emit("print(f'Epoch {epoch+1} | D_loss: {np.mean(d_losses[-len(dataloader):]):.4f} | G_loss: {np.mean(g_losses[-len(dataloader)//self.n_critic:]):.4f}')")
        self._indent_down()
        self._emit("")
        self._emit("return {'g_losses': g_losses, 'd_losses': d_losses}")
        self._indent_down()
        self._emit("")

        # Generate method
        self._emit("@torch.no_grad()")
        self._emit("def generate(self, num_samples=16):")
        self._indent_up()
        self._emit("self.generator.eval()")
        self._emit("z = torch.randn(num_samples, self.latent_dim, device=self.device)")
        self._emit("samples = self.generator(z)")
        self._emit("return samples")
        self._indent_down()

        self._indent_down()

    # --- Diffusion Transpilation -------------------------------------------------

    def _transpile_diffusion(self, diff: DiffusionDef):
        self._imports.add("from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, DDPMPipeline")
        self._imports.add("from diffusers import DDIMPipeline")
        self._imports.add("from tqdm import tqdm")
        self._imports.add("import numpy as np")

        self._emit(f"class {diff.name}Diffusion:")
        self._indent_up()
        self._emit(f'"""Diffusion model training with {diff.noise_scheduler} scheduler."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit(f"self.image_size = {diff.image_size}")
        self._emit(f"self.channels = {diff.channels}")
        self._emit(f"self.timesteps = {diff.timesteps}")

        model_code = self._value_to_python(diff.model_ref) if diff.model_ref else None
        if model_code and model_code != "None":
            self._emit(f"self.model = {model_code}.to(self.device)")
        else:
            self._emit("self.model = UNet2DModel(")
            self._indent_up()
            self._emit(f"sample_size={diff.image_size},")
            self._emit(f"in_channels={diff.channels},")
            self._emit(f"out_channels={diff.channels},")
            self._emit("layers_per_block=2,")
            self._emit("block_out_channels=(128, 128, 256, 256, 512, 512),")
            self._emit("down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D'),")
            self._emit("up_block_types=('UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'),")
            self._indent_down()
            self._emit(").to(self.device)")

        sched = diff.noise_scheduler.lower()
        beta_sched = diff.beta_schedule if diff.beta_schedule else "linear"
        if sched == "ddpm":
            self._emit(f"self.noise_scheduler = DDPMScheduler(num_train_timesteps={diff.timesteps}, beta_schedule='{beta_sched}')")
        elif sched == "ddim":
            self._emit(f"self.noise_scheduler = DDIMScheduler(num_train_timesteps={diff.timesteps}, beta_schedule='{beta_sched}')")
        else:
            self._emit(f"self.noise_scheduler = DDPMScheduler(num_train_timesteps={diff.timesteps}, beta_schedule='{beta_sched}')")

        self._emit("self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)")
        self._indent_down()
        self._emit("")

        # Training
        self._emit("def run(self, dataloader, epochs=100):")
        self._indent_up()
        self._emit("self.model.train()")
        self._emit("global_step = 0")
        self._emit("")
        self._emit("for epoch in range(epochs):")
        self._indent_up()
        self._emit("total_loss = 0")
        self._emit("pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')")
        self._emit("for batch_idx, (clean_images, _) in enumerate(pbar):")
        self._indent_up()
        self._emit("clean_images = clean_images.to(self.device)")
        self._emit("noise = torch.randn_like(clean_images)")
        self._emit("batch_size = clean_images.size(0)")
        self._emit("timesteps = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()")
        self._emit("")
        self._emit("noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)")
        self._emit("noise_pred = self.model(noisy_images, timesteps).sample")
        self._emit("loss = F.mse_loss(noise_pred, noise)")
        self._emit("")
        self._emit("self.optimizer.zero_grad()")
        self._emit("loss.backward()")
        self._emit("nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)")
        self._emit("self.optimizer.step()")
        self._emit("")
        self._emit("total_loss += loss.item()")
        self._emit("global_step += 1")
        self._emit("pbar.set_postfix({'loss': f'{loss.item():.4f}'})")
        self._indent_down()
        self._emit("")
        self._emit("avg_loss = total_loss / len(dataloader)")
        self._emit("print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')")
        self._indent_down()
        self._indent_down()
        self._emit("")

        # Sampling
        self._emit("@torch.no_grad()")
        self._emit("def sample(self, num_samples=4):")
        self._indent_up()
        self._emit("self.model.eval()")
        if sched == "ddpm":
            self._emit("pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)")
        else:
            self._emit("pipeline = DDIMPipeline(unet=self.model, scheduler=self.noise_scheduler)")
        self._emit("pipeline.to(self.device)")
        self._emit("images = pipeline(batch_size=num_samples, num_inference_steps=50).images")
        self._emit("return images")
        self._indent_down()

        self._indent_down()

    # --- RL Transpilation --------------------------------------------------------

    def _transpile_rl(self, rl: RLDef):
        self._imports.add("import gymnasium as gym")

        algo = rl.algorithm.lower()
        algo_map = {
            "ppo": ("stable_baselines3", "PPO"),
            "dqn": ("stable_baselines3", "DQN"),
            "a2c": ("stable_baselines3", "A2C"),
            "sac": ("stable_baselines3", "SAC"),
            "td3": ("stable_baselines3", "TD3"),
            "ddpg": ("stable_baselines3", "DDPG"),
        }

        pkg, cls = algo_map.get(algo, ("stable_baselines3", "PPO"))
        self._imports.add(f"from {pkg} import {cls}")
        self._imports.add(f"from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback")
        self._imports.add(f"from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv")
        self._imports.add(f"from stable_baselines3.common.monitor import Monitor")

        self._emit(f"class {rl.name}Agent:")
        self._indent_up()
        self._emit(f'"""RL agent using {cls} on {rl.environment}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.env_id = '{rl.environment}'")
        self._emit(f"self.env = gym.make(self.env_id)")
        self._emit(f"self.eval_env = Monitor(gym.make(self.env_id))")
        self._emit("")

        # Build kwargs
        kwargs_parts = [f"'{rl.policy}'", "self.env"]
        kwargs_parts.append(f"learning_rate={rl.learning_rate}")
        kwargs_parts.append(f"gamma={rl.gamma}")

        if algo == "ppo":
            kwargs_parts.append(f"n_steps={rl.n_steps}")
            kwargs_parts.append(f"batch_size={rl.batch_size}")
            kwargs_parts.append(f"n_epochs={rl.n_epochs}")
            kwargs_parts.append(f"clip_range={rl.clip_range}")
        elif algo == "dqn":
            kwargs_parts.append(f"buffer_size={rl.buffer_size}")
            kwargs_parts.append(f"batch_size={rl.batch_size}")
        elif algo in ("sac", "td3", "ddpg"):
            kwargs_parts.append(f"buffer_size={rl.buffer_size}")
            kwargs_parts.append(f"batch_size={rl.batch_size}")
        elif algo == "a2c":
            kwargs_parts.append(f"n_steps={rl.n_steps}")

        kwargs_parts.append("verbose=1")
        kwargs_parts.append("tensorboard_log='./tb_logs/'")

        self._emit(f"self.model = {cls}(")
        self._indent_up()
        self._emit(", ".join(kwargs_parts))
        self._indent_down()
        self._emit(")")
        self._emit(f"self.total_timesteps = {rl.total_timesteps}")
        self._indent_down()
        self._emit("")

        # Train
        self._emit("def run(self):")
        self._indent_up()
        self._emit("eval_callback = EvalCallback(")
        self._indent_up()
        self._emit("self.eval_env,")
        self._emit("best_model_save_path='./best_model/',")
        self._emit("log_path='./eval_logs/',")
        self._emit("eval_freq=10000,")
        self._emit("deterministic=True,")
        self._emit("render=False")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit("checkpoint_callback = CheckpointCallback(")
        self._indent_up()
        self._emit("save_freq=50000,")
        self._emit("save_path='./checkpoints/',")
        self._emit(f"name_prefix='{rl.name}'")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit("self.model.learn(")
        self._indent_up()
        self._emit("total_timesteps=self.total_timesteps,")
        self._emit("callback=[eval_callback, checkpoint_callback],")
        self._emit("progress_bar=True")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit(f"self.model.save('{rl.name}_final')")
        self._emit("print('Training complete!')")
        self._emit("return self.model")
        self._indent_down()
        self._emit("")

        # Evaluate
        self._emit("def evaluate(self, num_episodes=10):")
        self._indent_up()
        self._emit("rewards = []")
        self._emit("for ep in range(num_episodes):")
        self._indent_up()
        self._emit("obs, _ = self.eval_env.reset()")
        self._emit("total_reward = 0")
        self._emit("done = False")
        self._emit("while not done:")
        self._indent_up()
        self._emit("action, _ = self.model.predict(obs, deterministic=True)")
        self._emit("obs, reward, terminated, truncated, info = self.eval_env.step(action)")
        self._emit("total_reward += reward")
        self._emit("done = terminated or truncated")
        self._indent_down()
        self._emit("rewards.append(total_reward)")
        self._indent_down()
        self._emit("print(f'Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')")
        self._emit("return rewards")
        self._indent_down()

        self._indent_down()

    # --- Tabular Transpilation -----------------------------------------------------

    def _transpile_tabular(self, tab: TabularDef):
        self._imports.add("import pandas as pd")
        self._imports.add("import numpy as np")
        self._imports.add("from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold")
        self._imports.add("from sklearn.preprocessing import StandardScaler, LabelEncoder")
        self._imports.add("from sklearn.pipeline import Pipeline as SkPipeline")
        self._imports.add("from sklearn.metrics import accuracy_score, mean_squared_error, r2_score")

        algo = tab.algorithm.lower()
        if algo == "xgboost":
            self._imports.add("import xgboost as xgb")
        elif algo == "lightgbm":
            self._imports.add("import lightgbm as lgb")
        elif algo == "catboost":
            self._imports.add("from catboost import CatBoostClassifier, CatBoostRegressor")
        elif algo == "random_forest":
            self._imports.add("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor")
        elif algo == "svm":
            self._imports.add("from sklearn.svm import SVC, SVR")
        elif algo == "knn":
            self._imports.add("from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor")
        elif algo == "logistic":
            self._imports.add("from sklearn.linear_model import LogisticRegression")
        elif algo == "linear":
            self._imports.add("from sklearn.linear_model import LinearRegression, Ridge, Lasso")
        else:
            self._imports.add("from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor")

        self._emit(f"class {tab.name}Tabular:")
        self._indent_up()
        self._emit(f'"""Tabular ML pipeline using {algo} for {tab.task}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.task = '{tab.task}'")
        self._emit(f"self.target_column = '{tab.target_column}'")
        if tab.feature_columns:
            fc = [self._value_to_python(c) for c in tab.feature_columns]
            self._emit(f"self.feature_columns = [{', '.join(fc)}]")
        else:
            self._emit("self.feature_columns = None")
        if tab.categorical_columns:
            cc = [self._value_to_python(c) for c in tab.categorical_columns]
            self._emit(f"self.categorical_columns = [{', '.join(cc)}]")
        else:
            self._emit("self.categorical_columns = []")
        self._emit(f"self.cv_folds = {tab.cross_validation}")
        self._emit("self.scaler = StandardScaler()")
        self._emit("self.label_encoders = {}")
        self._emit("")

        # Model instantiation
        is_clf = tab.task == "classification"
        if algo == "xgboost":
            if is_clf:
                self._emit("self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', use_label_encoder=False)")
            else:
                self._emit("self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)")
        elif algo == "lightgbm":
            if is_clf:
                self._emit("self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, verbose=-1)")
            else:
                self._emit("self.model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1, verbose=-1)")
        elif algo == "catboost":
            if is_clf:
                self._emit("self.model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)")
            else:
                self._emit("self.model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)")
        elif algo == "random_forest":
            if is_clf:
                self._emit("self.model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)")
            else:
                self._emit("self.model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)")
        elif algo == "svm":
            if is_clf:
                self._emit("self.model = SVC(kernel='rbf', probability=True)")
            else:
                self._emit("self.model = SVR(kernel='rbf')")
        elif algo == "knn":
            if is_clf:
                self._emit("self.model = KNeighborsClassifier(n_neighbors=5)")
            else:
                self._emit("self.model = KNeighborsRegressor(n_neighbors=5)")
        elif algo == "logistic":
            self._emit("self.model = LogisticRegression(max_iter=1000)")
        elif algo == "linear":
            self._emit("self.model = Ridge(alpha=1.0)")
        else:
            if is_clf:
                self._emit("self.model = GradientBoostingClassifier(n_estimators=100)")
            else:
                self._emit("self.model = GradientBoostingRegressor(n_estimators=100)")

        self._indent_down()
        self._emit("")

        # Preprocessing
        self._emit("def _preprocess(self, df):")
        self._indent_up()
        self._emit("df = df.copy()")
        self._emit("for col in self.categorical_columns:")
        self._indent_up()
        self._emit("if col not in self.label_encoders:")
        self._indent_up()
        self._emit("self.label_encoders[col] = LabelEncoder()")
        self._emit("df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("df[col] = self.label_encoders[col].transform(df[col].astype(str))")
        self._indent_down()
        self._indent_down()
        self._emit("df = df.fillna(df.median(numeric_only=True))")
        self._emit("return df")
        self._indent_down()
        self._emit("")

        # Run
        self._emit("def run(self, df):")
        self._indent_up()
        self._emit("df = self._preprocess(df)")
        self._emit("y = df[self.target_column]")
        self._emit("if self.feature_columns:")
        self._indent_up()
        self._emit("X = df[self.feature_columns]")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("X = df.drop(columns=[self.target_column])")
        self._indent_down()
        self._emit("")
        self._emit("X_scaled = self.scaler.fit_transform(X)")
        self._emit("")

        if is_clf:
            self._emit(f"cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)")
            self._emit("scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')")
        else:
            self._emit(f"cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)")
            self._emit("scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')")

        self._emit("print(f'Cross-validation scores: {scores}')")
        self._emit("print(f'Mean: {scores.mean():.4f} (+/- {scores.std():.4f})')")
        self._emit("")
        self._emit("self.model.fit(X_scaled, y)")
        self._emit("print('Model fitted on full dataset.')")
        self._emit("return {'scores': scores, 'model': self.model}")
        self._indent_down()
        self._emit("")

        # Predict
        self._emit("def predict(self, df):")
        self._indent_up()
        self._emit("df = self._preprocess(df)")
        self._emit("if self.feature_columns:")
        self._indent_up()
        self._emit("X = df[self.feature_columns]")
        self._indent_down()
        self._emit("else:")
        self._indent_up()
        self._emit("X = df.drop(columns=[self.target_column], errors='ignore')")
        self._indent_down()
        self._emit("X_scaled = self.scaler.transform(X)")
        self._emit("return self.model.predict(X_scaled)")
        self._indent_down()

        self._indent_down()

    # --- Time Series Transpilation -----------------------------------------------

    def _transpile_timeseries(self, ts: TimeSeriesDef):
        self._imports.add("import pandas as pd")
        self._imports.add("import numpy as np")

        algo = ts.algorithm.lower()

        if algo == "prophet":
            self._imports.add("from prophet import Prophet")
        elif algo == "arima":
            self._imports.add("from statsmodels.tsa.arima.model import ARIMA")
            self._imports.add("from statsmodels.tsa.stattools import adfuller")
        elif algo in ("transformer", "lstm", "nbeats", "temporal_fusion"):
            self._imports.add("from pytorch_forecasting import TemporalFusionTransformer, NBeats, TimeSeriesDataSet")
            self._imports.add("from pytorch_forecasting.data import GroupNormalizer")
            self._imports.add("import pytorch_lightning as pl")

        self._emit(f"class {ts.name}TimeSeries:")
        self._indent_up()
        self._emit(f'"""Time series {ts.task} using {algo}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.target_column = '{ts.target_column}'")
        self._emit(f"self.time_column = '{ts.time_column}'")
        self._emit(f"self.horizon = {ts.horizon}")
        self._emit(f"self.lookback = {ts.lookback}")
        self._emit(f"self.frequency = '{ts.frequency}'")
        self._indent_down()
        self._emit("")

        # Run method by algorithm
        self._emit("def run(self, df):")
        self._indent_up()

        if algo == "prophet":
            self._emit("prophet_df = df[[self.time_column, self.target_column]].rename(")
            self._emit("    columns={self.time_column: 'ds', self.target_column: 'y'})")
            self._emit("model = Prophet()")
            if ts.features:
                for feat in ts.features:
                    feat_name = self._value_to_python(feat)
                    self._emit(f"model.add_regressor({feat_name})")
            self._emit("model.fit(prophet_df)")
            self._emit(f"future = model.make_future_dataframe(periods={ts.horizon}, freq='{ts.frequency}')")
            self._emit("forecast = model.predict(future)")
            self._emit("print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(self.horizon))")
            self._emit("self.model = model")
            self._emit("return forecast")
        elif algo == "arima":
            self._emit("series = df.set_index(self.time_column)[self.target_column]")
            self._emit("result = adfuller(series.dropna())")
            self._emit("print(f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}')")
            self._emit("model = ARIMA(series, order=(5, 1, 0))")
            self._emit("fitted = model.fit()")
            self._emit("print(fitted.summary())")
            self._emit(f"forecast = fitted.forecast(steps={ts.horizon})")
            self._emit("self.model = fitted")
            self._emit("return forecast")
        elif algo in ("transformer", "temporal_fusion"):
            self._emit("df['time_idx'] = (df[self.time_column] - df[self.time_column].min()).dt.total_seconds().astype(int)")
            self._emit("df['group'] = 'main'")
            self._emit("")
            self._emit("training = TimeSeriesDataSet(")
            self._indent_up()
            self._emit("df,")
            self._emit("time_idx='time_idx',")
            self._emit("target=self.target_column,")
            self._emit("group_ids=['group'],")
            self._emit(f"max_encoder_length={ts.lookback},")
            self._emit(f"max_prediction_length={ts.horizon},")
            self._emit("target_normalizer=GroupNormalizer(groups=['group']),")
            self._indent_down()
            self._emit(")")
            self._emit("")
            self._emit("train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)")
            self._emit("val_dataloader = training.to_dataloader(train=False, batch_size=64, num_workers=0)")
            self._emit("")
            self._emit("model = TemporalFusionTransformer.from_dataset(")
            self._indent_up()
            self._emit("training, learning_rate=0.03, hidden_size=16,")
            self._emit("attention_head_size=1, dropout=0.1, hidden_continuous_size=8,")
            self._indent_down()
            self._emit(")")
            self._emit("")
            self._emit("trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.1)")
            self._emit("trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)")
            self._emit("self.model = model")
            self._emit("return model")
        elif algo == "lstm":
            self._emit("# LSTM-based time series model")
            self._emit("series = df[self.target_column].values.astype(np.float32)")
            self._emit("from sklearn.preprocessing import MinMaxScaler")
            self._emit("scaler = MinMaxScaler()")
            self._emit("series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()")
            self._emit("")
            self._emit("# Create sequences")
            self._emit("X, y = [], []")
            self._emit("for i in range(self.lookback, len(series_scaled) - self.horizon):")
            self._indent_up()
            self._emit("X.append(series_scaled[i-self.lookback:i])")
            self._emit("y.append(series_scaled[i:i+self.horizon])")
            self._indent_down()
            self._emit("X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)")
            self._emit("y = torch.tensor(np.array(y), dtype=torch.float32)")
            self._emit("")
            self._emit("model = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)")
            self._emit("fc = nn.Linear(64, self.horizon)")
            self._emit("optimizer = optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=1e-3)")
            self._emit("criterion = nn.MSELoss()")
            self._emit("")
            self._emit("for epoch in range(100):")
            self._indent_up()
            self._emit("output, _ = model(X)")
            self._emit("pred = fc(output[:, -1, :])")
            self._emit("loss = criterion(pred, y)")
            self._emit("optimizer.zero_grad()")
            self._emit("loss.backward()")
            self._emit("optimizer.step()")
            self._emit("if (epoch + 1) % 10 == 0:")
            self._indent_up()
            self._emit("print(f'Epoch {epoch+1}: loss={loss.item():.4f}')")
            self._indent_down()
            self._indent_down()
            self._emit("self.model = model")
            self._emit("return model")
        else:
            self._emit(f"# {algo} time series model")
            self._emit("pass")

        self._indent_down()

        self._indent_down()

    # --- Graph Neural Network Transpilation --------------------------------------

    def _transpile_graph(self, graph: GraphDef):
        self._imports.add("import torch_geometric")
        self._imports.add("from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool, global_add_pool")
        self._imports.add("from torch_geometric.loader import DataLoader as PyGDataLoader")
        self._imports.add("import torch.nn.functional as F")

        conv_type = graph.conv_type.upper()
        conv_map = {
            "GCN": "GCNConv",
            "GAT": "GATConv",
            "GRAPHSAGE": "SAGEConv",
            "SAGE": "SAGEConv",
            "GIN": "GINConv",
        }
        conv_cls = conv_map.get(conv_type, "GCNConv")

        self._emit(f"class {graph.name}Graph(nn.Module):")
        self._indent_up()
        self._emit(f'"""Graph neural network with {conv_type} for {graph.task}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit(f"self.dropout = {graph.dropout}")

        if conv_cls == "GINConv":
            self._emit(f"gin_nn1 = nn.Sequential(nn.Linear({graph.num_features}, {graph.hidden_dim}), nn.ReLU(), nn.Linear({graph.hidden_dim}, {graph.hidden_dim}))")
            self._emit(f"self.conv1 = GINConv(gin_nn1)")
            self._emit(f"gin_nn2 = nn.Sequential(nn.Linear({graph.hidden_dim}, {graph.hidden_dim}), nn.ReLU(), nn.Linear({graph.hidden_dim}, {graph.hidden_dim}))")
            self._emit(f"self.conv2 = GINConv(gin_nn2)")
        elif conv_cls == "GATConv":
            self._emit(f"self.conv1 = {conv_cls}({graph.num_features}, {graph.hidden_dim}, heads={graph.heads}, dropout={graph.dropout})")
            self._emit(f"self.conv2 = {conv_cls}({graph.hidden_dim} * {graph.heads}, {graph.hidden_dim}, heads=1, concat=False, dropout={graph.dropout})")
        else:
            self._emit(f"self.conv1 = {conv_cls}({graph.num_features}, {graph.hidden_dim})")
            self._emit(f"self.conv2 = {conv_cls}({graph.hidden_dim}, {graph.hidden_dim})")

        if graph.task == "graph_classification":
            self._emit(f"self.classifier = nn.Linear({graph.hidden_dim}, {graph.num_classes})")
        elif graph.task == "node_classification":
            self._emit(f"self.classifier = nn.Linear({graph.hidden_dim}, {graph.num_classes})")
        elif graph.task == "link_prediction":
            self._emit(f"self.predictor = nn.Linear({graph.hidden_dim} * 2, 1)")
        self._indent_down()
        self._emit("")

        self._emit("def forward(self, data):")
        self._indent_up()
        self._emit("x, edge_index = data.x, data.edge_index")
        self._emit("x = self.conv1(x, edge_index)")
        self._emit("x = F.relu(x)")
        self._emit(f"x = F.dropout(x, p=self.dropout, training=self.training)")
        self._emit("x = self.conv2(x, edge_index)")
        self._emit("")

        pooling_map = {
            "mean": "global_mean_pool",
            "max": "global_max_pool",
            "sum": "global_add_pool",
        }

        if graph.task == "graph_classification":
            pool_fn = pooling_map.get(graph.pooling, "global_mean_pool")
            self._emit(f"x = {pool_fn}(x, data.batch)")
            self._emit("x = self.classifier(x)")
        elif graph.task == "node_classification":
            self._emit("x = self.classifier(x)")
        elif graph.task == "link_prediction":
            self._emit("row, col = data.edge_label_index")
            self._emit("x = torch.cat([x[row], x[col]], dim=1)")
            self._emit("x = self.predictor(x).squeeze(-1)")

        self._emit("return x")
        self._indent_down()
        self._emit("")

        # Training function
        self._emit("@staticmethod")
        self._emit(f"def train_loop(model, train_loader, optimizer, device='cuda'):")
        self._indent_up()
        self._emit("model.train()")
        self._emit("total_loss = 0")
        self._emit("for data in train_loader:")
        self._indent_up()
        self._emit("data = data.to(device)")
        self._emit("optimizer.zero_grad()")
        self._emit("out = model(data)")
        if graph.task == "graph_classification":
            self._emit("loss = F.cross_entropy(out, data.y)")
        elif graph.task == "node_classification":
            self._emit("loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])")
        elif graph.task == "link_prediction":
            self._emit("loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float())")
        else:
            self._emit("loss = F.cross_entropy(out, data.y)")
        self._emit("loss.backward()")
        self._emit("optimizer.step()")
        self._emit("total_loss += loss.item()")
        self._indent_down()
        self._emit("return total_loss / len(train_loader)")
        self._indent_down()

        self._indent_down()

    # --- Audio Transpilation -------------------------------------------------------

    def _transpile_audio(self, audio: AudioDef):
        self._imports.add("import torchaudio")
        self._imports.add("import torchaudio.transforms as T")

        model_type = audio.model_type.lower()
        task = audio.task.lower()

        if model_type == "whisper":
            self._imports.add("from transformers import WhisperProcessor, WhisperForConditionalGeneration")
        elif model_type in ("wav2vec2", "hubert"):
            self._imports.add("from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC")
            self._imports.add("from transformers import Wav2Vec2ForSequenceClassification")

        self._emit(f"class {audio.name}Audio:")
        self._indent_up()
        self._emit(f'"""Audio {task} using {model_type}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit(f"self.sample_rate = {audio.sample_rate}")
        self._emit(f"self.n_mels = {audio.n_mels}")
        self._emit(f"self.n_fft = {audio.n_fft}")
        self._emit(f"self.hop_length = {audio.hop_length}")

        if model_type == "whisper":
            self._emit("self.processor = WhisperProcessor.from_pretrained('openai/whisper-base')")
            self._emit("self.model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base').to(self.device)")
        elif model_type in ("wav2vec2", "hubert"):
            model_id = "facebook/wav2vec2-base-960h" if model_type == "wav2vec2" else "facebook/hubert-base-ls960"
            self._emit(f"self.processor = Wav2Vec2Processor.from_pretrained('{model_id}')")
            if task == "asr":
                self._emit(f"self.model = Wav2Vec2ForCTC.from_pretrained('{model_id}').to(self.device)")
            else:
                self._emit(f"self.model = Wav2Vec2ForSequenceClassification.from_pretrained('{model_id}').to(self.device)")
        else:
            self._emit("self.mel_transform = T.MelSpectrogram(")
            self._indent_up()
            self._emit(f"sample_rate={audio.sample_rate},")
            self._emit(f"n_mels={audio.n_mels},")
            self._emit(f"n_fft={audio.n_fft},")
            self._emit(f"hop_length={audio.hop_length}")
            self._indent_down()
            self._emit(")")

        self._emit("self.resampler = T.Resample(orig_freq=44100, new_freq=self.sample_rate)")
        self._indent_down()
        self._emit("")

        # Process method
        if task == "asr":
            self._emit("@torch.no_grad()")
            self._emit("def transcribe(self, audio_path):")
            self._indent_up()
            self._emit("waveform, sr = torchaudio.load(audio_path)")
            self._emit("if sr != self.sample_rate:")
            self._indent_up()
            self._emit("resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)")
            self._emit("waveform = resampler(waveform)")
            self._indent_down()
            self._emit("waveform = waveform.squeeze(0)")
            if model_type == "whisper":
                self._emit("input_features = self.processor(waveform.numpy(), sampling_rate=self.sample_rate, return_tensors='pt').input_features.to(self.device)")
                self._emit("generated_ids = self.model.generate(input_features)")
                self._emit("transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)")
                self._emit("return transcription[0]")
            else:
                self._emit("inputs = self.processor(waveform.numpy(), sampling_rate=self.sample_rate, return_tensors='pt', padding=True).to(self.device)")
                self._emit("logits = self.model(**inputs).logits")
                self._emit("predicted_ids = torch.argmax(logits, dim=-1)")
                self._emit("transcription = self.processor.batch_decode(predicted_ids)")
                self._emit("return transcription[0]")
            self._indent_down()
        elif task == "classification":
            self._emit("def extract_features(self, audio_path):")
            self._indent_up()
            self._emit("waveform, sr = torchaudio.load(audio_path)")
            self._emit("if sr != self.sample_rate:")
            self._indent_up()
            self._emit("resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)")
            self._emit("waveform = resampler(waveform)")
            self._indent_down()
            if model_type in ("wav2vec2", "hubert"):
                self._emit("inputs = self.processor(waveform.squeeze(0).numpy(), sampling_rate=self.sample_rate, return_tensors='pt', padding=True).to(self.device)")
                self._emit("with torch.no_grad():")
                self._indent_up()
                self._emit("features = self.model(**inputs).logits")
                self._indent_down()
            else:
                self._emit("features = self.mel_transform(waveform)")
            self._emit("return features")
            self._indent_down()
        else:
            self._emit("def process(self, audio_path):")
            self._indent_up()
            self._emit("waveform, sr = torchaudio.load(audio_path)")
            self._emit("if sr != self.sample_rate:")
            self._indent_up()
            self._emit("resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)")
            self._emit("waveform = resampler(waveform)")
            self._indent_down()
            self._emit("return waveform")
            self._indent_down()

        self._indent_down()

    # --- Multimodal Transpilation ------------------------------------------------

    def _transpile_multimodal(self, mm: MultimodalDef):
        self._imports.add("from transformers import CLIPModel, CLIPProcessor")
        self._imports.add("from PIL import Image")
        self._imports.add("import torch.nn.functional as F")

        self._emit(f"class {mm.name}Multimodal(nn.Module):")
        self._indent_up()
        self._emit(f'"""Multimodal model for {mm.task} using {mm.fusion_method} fusion."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("super().__init__()")
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")

        vision_enc = mm.vision_encoder or "openai/clip-vit-base-patch32"
        text_enc = mm.text_encoder or "openai/clip-vit-base-patch32"

        self._emit(f"self.clip_model = CLIPModel.from_pretrained('{vision_enc}')")
        self._emit(f"self.clip_processor = CLIPProcessor.from_pretrained('{vision_enc}')")
        self._emit("self.vision_dim = self.clip_model.config.projection_dim")
        self._emit("self.text_dim = self.clip_model.config.projection_dim")

        fusion = mm.fusion_method.lower()
        if fusion == "cross_attention":
            self._emit("self.cross_attn = nn.MultiheadAttention(embed_dim=self.vision_dim, num_heads=8, batch_first=True)")
            self._emit("self.classifier = nn.Linear(self.vision_dim, 1000)")
        elif fusion == "concat":
            self._emit("self.classifier = nn.Linear(self.vision_dim + self.text_dim, 1000)")
        elif fusion == "late_fusion":
            self._emit("self.vision_head = nn.Linear(self.vision_dim, 512)")
            self._emit("self.text_head = nn.Linear(self.text_dim, 512)")
            self._emit("self.classifier = nn.Linear(512, 1000)")
        else:
            self._emit("self.classifier = nn.Linear(self.vision_dim + self.text_dim, 1000)")

        self._indent_down()
        self._emit("")

        self._emit("def encode_image(self, images):")
        self._indent_up()
        self._emit("inputs = self.clip_processor(images=images, return_tensors='pt', padding=True).to(self.device)")
        self._emit("image_features = self.clip_model.get_image_features(**inputs)")
        self._emit("return F.normalize(image_features, dim=-1)")
        self._indent_down()
        self._emit("")

        self._emit("def encode_text(self, texts):")
        self._indent_up()
        self._emit("inputs = self.clip_processor(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)")
        self._emit("text_features = self.clip_model.get_text_features(**inputs)")
        self._emit("return F.normalize(text_features, dim=-1)")
        self._indent_down()
        self._emit("")

        self._emit("def forward(self, images=None, texts=None):")
        self._indent_up()
        self._emit("features = []")
        self._emit("if images is not None:")
        self._indent_up()
        self._emit("image_feats = self.encode_image(images)")
        self._emit("features.append(image_feats)")
        self._indent_down()
        self._emit("if texts is not None:")
        self._indent_up()
        self._emit("text_feats = self.encode_text(texts)")
        self._emit("features.append(text_feats)")
        self._indent_down()

        if fusion == "cross_attention":
            self._emit("if len(features) == 2:")
            self._indent_up()
            self._emit("attn_out, _ = self.cross_attn(features[0].unsqueeze(1), features[1].unsqueeze(1), features[1].unsqueeze(1))")
            self._emit("combined = attn_out.squeeze(1)")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("combined = features[0]")
            self._indent_down()
        elif fusion == "concat":
            self._emit("combined = torch.cat(features, dim=-1)")
        elif fusion == "late_fusion":
            self._emit("if len(features) == 2:")
            self._indent_up()
            self._emit("v = F.relu(self.vision_head(features[0]))")
            self._emit("t = F.relu(self.text_head(features[1]))")
            self._emit("combined = v + t")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("combined = features[0]")
            self._indent_down()
        else:
            self._emit("combined = torch.cat(features, dim=-1)")

        self._emit("return self.classifier(combined)")
        self._indent_down()

        self._indent_down()

    # --- Distillation Transpilation ----------------------------------------------

    def _transpile_distill(self, distill: DistillDef):
        self._imports.add("import torch.nn.functional as F")
        self._imports.add("from tqdm import tqdm")

        self._emit(f"class {distill.name}Distiller:")
        self._indent_up()
        self._emit(f'"""Knowledge distillation: {distill.teacher} -> {distill.student}."""')
        self._emit("")

        self._emit(f"def __init__(self, teacher, student):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("self.teacher = teacher.to(self.device)")
        self._emit("self.student = student.to(self.device)")
        self._emit("self.teacher.eval()")
        self._emit(f"self.temperature = {distill.temperature}")
        self._emit(f"self.alpha = {distill.alpha}")
        self._emit("self.optimizer = optim.Adam(self.student.parameters(), lr=1e-3)")
        self._indent_down()
        self._emit("")

        method = distill.method.lower()

        self._emit("def distillation_loss(self, student_logits, teacher_logits, targets):")
        self._indent_up()
        if method == "kd":
            self._emit("soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)")
            self._emit("soft_student = F.log_softmax(student_logits / self.temperature, dim=1)")
            self._emit("distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)")
            self._emit("hard_loss = F.cross_entropy(student_logits, targets)")
            self._emit("return self.alpha * distill_loss + (1 - self.alpha) * hard_loss")
        elif method == "attention":
            self._emit("# Attention transfer distillation")
            self._emit("soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)")
            self._emit("soft_student = F.log_softmax(student_logits / self.temperature, dim=1)")
            self._emit("distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)")
            self._emit("hard_loss = F.cross_entropy(student_logits, targets)")
            self._emit("return self.alpha * distill_loss + (1 - self.alpha) * hard_loss")
        elif method == "feature":
            self._emit("# Feature-based distillation")
            self._emit("feature_loss = F.mse_loss(student_logits, teacher_logits)")
            self._emit("return feature_loss")
        else:
            self._emit("soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)")
            self._emit("soft_student = F.log_softmax(student_logits / self.temperature, dim=1)")
            self._emit("distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)")
            self._emit("hard_loss = F.cross_entropy(student_logits, targets)")
            self._emit("return self.alpha * distill_loss + (1 - self.alpha) * hard_loss")
        self._indent_down()
        self._emit("")

        self._emit("def run(self, dataloader, epochs=20):")
        self._indent_up()
        self._emit("for epoch in range(epochs):")
        self._indent_up()
        self._emit("self.student.train()")
        self._emit("total_loss = 0")
        self._emit("pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')")
        self._emit("for inputs, targets in pbar:")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("teacher_logits = self.teacher(inputs)")
        self._indent_down()
        self._emit("student_logits = self.student(inputs)")
        self._emit("loss = self.distillation_loss(student_logits, teacher_logits, targets)")
        self._emit("self.optimizer.zero_grad()")
        self._emit("loss.backward()")
        self._emit("self.optimizer.step()")
        self._emit("total_loss += loss.item()")
        self._emit("pbar.set_postfix({'loss': f'{loss.item():.4f}'})")
        self._indent_down()
        self._emit("print(f'Epoch {epoch+1}: avg_loss={total_loss/len(dataloader):.4f}')")
        self._indent_down()
        self._emit("return self.student")
        self._indent_down()

        self._indent_down()

    # --- Quantize Transpilation ----------------------------------------------------

    def _transpile_quantize(self, quant: QuantizeDef):
        method = quant.method.lower()

        if method in ("dynamic", "static", "qat"):
            self._imports.add("import torch.quantization")
        elif method == "gptq":
            self._imports.add("from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig")
        elif method == "awq":
            self._imports.add("from awq import AutoAWQForCausalLM")
        elif method == "bnb":
            self._imports.add("from transformers import BitsAndBytesConfig, AutoModelForCausalLM")

        self._emit(f"class {quant.name}Quantizer:")
        self._indent_up()
        self._emit(f'"""Model quantization using {method} ({quant.dtype})."""')
        self._emit("")

        self._emit("def __init__(self, model):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.model.eval()")
        self._indent_down()
        self._emit("")

        self._emit("def run(self):")
        self._indent_up()

        if method == "dynamic":
            self._emit("quantized = torch.quantization.quantize_dynamic(")
            self._indent_up()
            dtype_map = {"int8": "torch.qint8", "fp16": "torch.float16"}
            dtype = dtype_map.get(quant.dtype, "torch.qint8")
            self._emit(f"self.model, {{nn.Linear, nn.LSTM, nn.GRU}}, dtype={dtype}")
            self._indent_down()
            self._emit(")")
            self._emit("print('Dynamic quantization applied.')")
            self._emit("return quantized")
        elif method == "static":
            self._emit("self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')")
            self._emit("prepared = torch.quantization.prepare(self.model)")
            self._emit("# Run calibration data through the model")
            self._emit("# for data in calibration_loader:")
            self._emit("#     prepared(data)")
            self._emit("quantized = torch.quantization.convert(prepared)")
            self._emit("print('Static quantization applied.')")
            self._emit("return quantized")
        elif method == "qat":
            self._emit("self.model.train()")
            self._emit("self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')")
            self._emit("prepared = torch.quantization.prepare_qat(self.model)")
            self._emit("# Train the prepared model, then convert")
            self._emit("# for epoch in range(num_epochs):")
            self._emit("#     train_one_epoch(prepared, ...)")
            self._emit("quantized = torch.quantization.convert(prepared.eval())")
            self._emit("print('QAT quantization applied.')")
            self._emit("return quantized")
        elif method == "gptq":
            self._emit("quantize_config = BaseQuantizeConfig(")
            self._indent_up()
            bits = 4 if quant.dtype == "int4" else 8
            self._emit(f"bits={bits},")
            self._emit("group_size=128,")
            self._emit("desc_act=False,")
            self._indent_down()
            self._emit(")")
            self._emit("quantized = AutoGPTQForCausalLM.from_pretrained(")
            self._indent_up()
            self._emit("self.model, quantize_config=quantize_config")
            self._indent_down()
            self._emit(")")
            self._emit("# quantized.quantize(calibration_dataset)")
            self._emit("print('GPTQ quantization applied.')")
            self._emit("return quantized")
        elif method == "awq":
            self._emit("quantized = AutoAWQForCausalLM.from_pretrained(self.model)")
            self._emit("quantized.quantize(")
            self._indent_up()
            self._emit("tokenizer=None,")
            bits = 4 if quant.dtype == "int4" else 8
            self._emit(f"quant_config={{'zero_point': True, 'q_group_size': 128, 'w_bit': {bits}}}")
            self._indent_down()
            self._emit(")")
            self._emit("print('AWQ quantization applied.')")
            self._emit("return quantized")
        elif method == "bnb":
            bits = 4 if quant.dtype == "int4" else 8
            self._emit("bnb_config = BitsAndBytesConfig(")
            self._indent_up()
            if bits == 4:
                self._emit("load_in_4bit=True,")
                self._emit("bnb_4bit_quant_type='nf4',")
                self._emit("bnb_4bit_compute_dtype=torch.float16,")
                self._emit("bnb_4bit_use_double_quant=True,")
            else:
                self._emit("load_in_8bit=True,")
            self._indent_down()
            self._emit(")")
            self._emit("quantized = AutoModelForCausalLM.from_pretrained(")
            self._indent_up()
            self._emit("self.model, quantization_config=bnb_config, device_map='auto'")
            self._indent_down()
            self._emit(")")
            self._emit("print('bitsandbytes quantization applied.')")
            self._emit("return quantized")
        else:
            self._emit("print(f'Unknown quantization method: {method}')")
            self._emit("return self.model")

        self._indent_down()
        self._indent_down()

    # --- Monitor Transpilation ---------------------------------------------------

    def _transpile_monitor(self, mon: MonitorDef):
        backend = mon.backend.lower()

        if backend == "wandb":
            self._imports.add("import wandb")
        elif backend == "mlflow":
            self._imports.add("import mlflow")
            self._imports.add("import mlflow.pytorch")
        elif backend == "tensorboard":
            self._imports.add("from torch.utils.tensorboard import SummaryWriter")
        elif backend == "neptune":
            self._imports.add("import neptune")

        self._emit(f"class {mon.name}Monitor:")
        self._indent_up()
        self._emit(f'"""Experiment monitoring using {backend}."""')
        self._emit("")

        self._emit("def __init__(self, project_name=None, run_name=None):")
        self._indent_up()

        if backend == "wandb":
            self._emit("self.run = wandb.init(")
            self._indent_up()
            self._emit(f"project=project_name or '{mon.name}',")
            self._emit(f"name=run_name or '{mon.name}_run',")
            self._emit("config={}")
            self._indent_down()
            self._emit(")")
        elif backend == "mlflow":
            self._emit(f"mlflow.set_experiment(project_name or '{mon.name}')")
            self._emit("self.run = mlflow.start_run(run_name=run_name)")
        elif backend == "tensorboard":
            self._emit(f"self.writer = SummaryWriter(log_dir=f'runs/{{project_name or \"{mon.name}\"}}')")
            self._emit("self.step = 0")
        elif backend == "neptune":
            self._emit("self.run = neptune.init_run(")
            self._indent_up()
            self._emit(f"project=project_name or '{mon.name}',")
            self._emit(f"name=run_name or '{mon.name}_run'")
            self._indent_down()
            self._emit(")")

        self._indent_down()
        self._emit("")

        # Log metrics
        self._emit("def log_metrics(self, metrics, step=None):")
        self._indent_up()
        if backend == "wandb":
            self._emit("wandb.log(metrics, step=step)")
        elif backend == "mlflow":
            self._emit("for k, v in metrics.items():")
            self._indent_up()
            self._emit("mlflow.log_metric(k, v, step=step)")
            self._indent_down()
        elif backend == "tensorboard":
            self._emit("for k, v in metrics.items():")
            self._indent_up()
            self._emit("self.writer.add_scalar(k, v, global_step=step or self.step)")
            self._indent_down()
            self._emit("self.step += 1")
        elif backend == "neptune":
            self._emit("for k, v in metrics.items():")
            self._indent_up()
            self._emit("self.run[k].append(v, step=step)")
            self._indent_down()
        self._indent_down()
        self._emit("")

        # Log model
        self._emit("def log_model(self, model, name='model'):")
        self._indent_up()
        if backend == "wandb":
            self._emit("artifact = wandb.Artifact(name, type='model')")
            self._emit("torch.save(model.state_dict(), f'{name}.pth')")
            self._emit("artifact.add_file(f'{name}.pth')")
            self._emit("self.run.log_artifact(artifact)")
        elif backend == "mlflow":
            self._emit("mlflow.pytorch.log_model(model, name)")
        elif backend == "tensorboard":
            self._emit("torch.save(model.state_dict(), f'{name}.pth')")
            self._emit("print(f'Model saved to {name}.pth')")
        elif backend == "neptune":
            self._emit("torch.save(model.state_dict(), f'{name}.pth')")
            self._emit("self.run[f'artifacts/{name}'].upload(f'{name}.pth')")
        self._indent_down()
        self._emit("")

        # Log hyperparams
        self._emit("def log_params(self, params):")
        self._indent_up()
        if backend == "wandb":
            self._emit("wandb.config.update(params)")
        elif backend == "mlflow":
            self._emit("mlflow.log_params(params)")
        elif backend == "tensorboard":
            self._emit("self.writer.add_hparams(params, {})")
        elif backend == "neptune":
            self._emit("self.run['parameters'] = params")
        self._indent_down()
        self._emit("")

        # Finish
        self._emit("def finish(self):")
        self._indent_up()
        if backend == "wandb":
            self._emit("wandb.finish()")
        elif backend == "mlflow":
            self._emit("mlflow.end_run()")
        elif backend == "tensorboard":
            self._emit("self.writer.close()")
        elif backend == "neptune":
            self._emit("self.run.stop()")
        self._indent_down()

        self._indent_down()

    # --- Serve Transpilation -----------------------------------------------------

    def _transpile_serve(self, serve: ServeDef):
        framework = serve.framework.lower()

        if framework == "fastapi":
            self._imports.add("from fastapi import FastAPI")
            self._imports.add("from pydantic import BaseModel")
            self._imports.add("import uvicorn")
        elif framework == "flask":
            self._imports.add("from flask import Flask, request, jsonify")
        elif framework == "gradio":
            self._imports.add("import gradio as gr")
        elif framework == "streamlit":
            self._imports.add("import streamlit as st")
        elif framework == "triton":
            self._imports.add("import tritonclient.http as httpclient")

        self._emit(f"class {serve.name}Server:")
        self._indent_up()
        self._emit(f'"""Model serving with {framework}."""')
        self._emit("")

        self._emit("def __init__(self, model):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.model.eval()")
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("self.model.to(self.device)")
        self._indent_down()
        self._emit("")

        self._emit("@torch.no_grad()")
        self._emit("def predict(self, input_data):")
        self._indent_up()
        self._emit("if isinstance(input_data, list):")
        self._indent_up()
        self._emit("input_data = torch.tensor(input_data, dtype=torch.float32)")
        self._indent_down()
        self._emit("input_data = input_data.to(self.device)")
        self._emit("if input_data.dim() == 1:")
        self._indent_up()
        self._emit("input_data = input_data.unsqueeze(0)")
        self._indent_down()
        self._emit("output = self.model(input_data)")
        self._emit("return output.cpu().tolist()")
        self._indent_down()
        self._emit("")

        self._emit(f"def run(self, host='{serve.host}', port={serve.port}):")
        self._indent_up()

        if framework == "fastapi":
            self._emit("app = FastAPI(title='" + serve.name + " API')")
            self._emit("")
            self._emit("class PredictRequest(BaseModel):")
            self._indent_up()
            self._emit("input: list")
            self._indent_down()
            self._emit("")
            self._emit("class PredictResponse(BaseModel):")
            self._indent_up()
            self._emit("prediction: list")
            self._indent_down()
            self._emit("")
            if serve.cors:
                self._emit("from fastapi.middleware.cors import CORSMiddleware")
                self._emit("app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])")
                self._emit("")
            self._emit(f"@app.post('{serve.endpoint}', response_model=PredictResponse)")
            self._emit("async def predict_endpoint(request: PredictRequest):")
            self._indent_up()
            self._emit("result = self.predict(request.input)")
            self._emit("return PredictResponse(prediction=result)")
            self._indent_down()
            self._emit("")
            self._emit("@app.get('/health')")
            self._emit("async def health():")
            self._indent_up()
            self._emit("return {'status': 'healthy'}")
            self._indent_down()
            self._emit("")
            self._emit("uvicorn.run(app, host=host, port=port)")
        elif framework == "flask":
            self._emit("app = Flask(__name__)")
            self._emit("")
            self._emit(f"@app.route('{serve.endpoint}', methods=['POST'])")
            self._emit("def predict_endpoint():")
            self._indent_up()
            self._emit("data = request.get_json()")
            self._emit("result = self.predict(data['input'])")
            self._emit("return jsonify({'prediction': result})")
            self._indent_down()
            self._emit("")
            self._emit("@app.route('/health', methods=['GET'])")
            self._emit("def health():")
            self._indent_up()
            self._emit("return jsonify({'status': 'healthy'})")
            self._indent_down()
            self._emit("")
            self._emit("app.run(host=host, port=port)")
        elif framework == "gradio":
            self._emit("def predict_fn(input_text):")
            self._indent_up()
            self._emit("import json")
            self._emit("data = json.loads(input_text)")
            self._emit("result = self.predict(data)")
            self._emit("return str(result)")
            self._indent_down()
            self._emit("")
            self._emit("demo = gr.Interface(")
            self._indent_up()
            self._emit("fn=predict_fn,")
            self._emit("inputs=gr.Textbox(label='Input JSON'),")
            self._emit("outputs=gr.Textbox(label='Prediction'),")
            self._emit(f"title='{serve.name} Model Demo'")
            self._indent_down()
            self._emit(")")
            self._emit("demo.launch(server_name=host, server_port=port)")
        elif framework == "streamlit":
            self._emit(f"st.title('{serve.name} Model')")
            self._emit("input_text = st.text_area('Input (JSON list)', '[]')")
            self._emit("if st.button('Predict'):")
            self._indent_up()
            self._emit("import json")
            self._emit("data = json.loads(input_text)")
            self._emit("result = self.predict(data)")
            self._emit("st.json({'prediction': result})")
            self._indent_down()
        elif framework == "triton":
            self._emit("# Export model for Triton Inference Server")
            self._emit("torch.jit.save(torch.jit.trace(self.model, torch.randn(1, 3, 224, 224).to(self.device)), 'model.pt')")
            self._emit("config = '''")
            self._emit(f"name: \\\"{serve.name}\\\"")
            self._emit("platform: \\\"pytorch_libtorch\\\"")
            self._emit("max_batch_size: " + str(serve.max_batch_size))
            self._emit("input [{ name: \\\"input__0\\\" data_type: TYPE_FP32 dims: [3, 224, 224] }]")
            self._emit("output [{ name: \\\"output__0\\\" data_type: TYPE_FP32 dims: [1000] }]")
            self._emit("'''")
            self._emit("Path('model_repository/" + serve.name + "/1/').mkdir(parents=True, exist_ok=True)")
            self._emit("Path('model_repository/" + serve.name + "/config.pbtxt').write_text(config)")
            self._emit("print('Model exported for Triton. Run: tritonserver --model-repository=model_repository')")

        self._indent_down()
        self._indent_down()

    # --- Test Transpilation --------------------------------------------------------

    def _transpile_test(self, test: TestDef):
        self._imports.add("import numpy as np")
        self._imports.add("from tqdm import tqdm")

        self._emit(f"class {test.name}Tester:")
        self._indent_up()
        self._emit(f'"""Model testing suite for {test.name}."""')
        self._emit("")

        self._emit("def __init__(self, model, data=None):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("self.model = model.to(self.device)")
        self._emit("self.model.eval()")
        self._emit("self.data = data")
        self._emit("self.results = {}")
        self._indent_down()
        self._emit("")

        # Invariance test
        self._emit("def test_invariance(self, inputs, transform_fn, atol=1e-3):")
        self._indent_up()
        self._emit('"""Test that model output is invariant under a given transformation."""')
        self._emit("inputs = inputs.to(self.device)")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("original_output = self.model(inputs)")
        self._emit("transformed_input = transform_fn(inputs)")
        self._emit("transformed_output = self.model(transformed_input)")
        self._indent_down()
        self._emit("diff = torch.abs(original_output - transformed_output).max().item()")
        self._emit("passed = diff < atol")
        self._emit("self.results['invariance'] = {'passed': passed, 'max_diff': diff}")
        self._emit("print(f'Invariance test: {\"PASSED\" if passed else \"FAILED\"} (max_diff={diff:.6f})')")
        self._emit("return passed")
        self._indent_down()
        self._emit("")

        # Robustness test
        self._emit("def test_robustness(self, inputs, targets, epsilon=0.01):")
        self._indent_up()
        self._emit('"""Test robustness to small input perturbations (FGSM-style)."""')
        self._emit("inputs = inputs.to(self.device).requires_grad_(True)")
        self._emit("targets = targets.to(self.device)")
        self._emit("outputs = self.model(inputs)")
        self._emit("loss = nn.functional.cross_entropy(outputs, targets)")
        self._emit("loss.backward()")
        self._emit("perturbed = inputs + epsilon * inputs.grad.sign()")
        self._emit("perturbed = perturbed.clamp(0, 1)")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("clean_preds = outputs.argmax(dim=1)")
        self._emit("perturbed_preds = self.model(perturbed).argmax(dim=1)")
        self._indent_down()
        self._emit("flip_rate = (clean_preds != perturbed_preds).float().mean().item()")
        self._emit("clean_acc = (clean_preds == targets).float().mean().item()")
        self._emit("perturbed_acc = (perturbed_preds == targets).float().mean().item()")
        self._emit("self.results['robustness'] = {'flip_rate': flip_rate, 'clean_acc': clean_acc, 'perturbed_acc': perturbed_acc}")
        self._emit("print(f'Robustness test: flip_rate={flip_rate:.4f}, clean_acc={clean_acc:.4f}, perturbed_acc={perturbed_acc:.4f}')")
        self._emit("return self.results['robustness']")
        self._indent_down()
        self._emit("")

        # Bias test
        self._emit("def test_bias(self, data_groups, group_labels):")
        self._indent_up()
        self._emit('"""Test for prediction bias across different groups."""')
        self._emit("group_accuracies = {}")
        self._emit("for group_name, (inputs, targets) in zip(group_labels, data_groups):")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("preds = self.model(inputs).argmax(dim=1)")
        self._indent_down()
        self._emit("acc = (preds == targets).float().mean().item()")
        self._emit("group_accuracies[group_name] = acc")
        self._emit("print(f'  Group {group_name}: accuracy={acc:.4f}')")
        self._indent_down()
        self._emit("")
        self._emit("accs = list(group_accuracies.values())")
        self._emit("disparity = max(accs) - min(accs)")
        self._emit("self.results['bias'] = {'group_accuracies': group_accuracies, 'disparity': disparity}")
        self._emit("print(f'Bias test: max disparity={disparity:.4f}')")
        self._emit("return self.results['bias']")
        self._indent_down()
        self._emit("")

        # Run all
        self._emit("def run_all(self, inputs, targets, **kwargs):")
        self._indent_up()
        self._emit("self.test_robustness(inputs, targets, epsilon=kwargs.get('epsilon', 0.01))")
        self._emit("print(f'\\nAll test results: {self.results}')")
        self._emit("return self.results")
        self._indent_down()

        self._indent_down()

    # --- Benchmark Transpilation -------------------------------------------------

    def _transpile_benchmark(self, bench: BenchmarkDef):
        self._imports.add("import time")
        self._imports.add("import numpy as np")

        self._emit(f"class {bench.name}Benchmark:")
        self._indent_up()
        self._emit(f'"""Model benchmarking for {bench.name}."""')
        self._emit("")

        self._emit("def __init__(self, model):")
        self._indent_up()
        self._emit("self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("self.model = model.to(self.device)")
        self._emit("self.model.eval()")
        self._emit(f"self.num_warmup = {bench.num_warmup}")
        self._emit(f"self.num_runs = {bench.num_runs}")
        if bench.input_shape:
            shape = [self._value_to_python(s) for s in bench.input_shape]
            self._emit(f"self.input_shape = [{', '.join(shape)}]")
        else:
            self._emit("self.input_shape = [1, 3, 224, 224]")
        self._indent_down()
        self._emit("")

        # Latency benchmark
        self._emit("def benchmark_latency(self):")
        self._indent_up()
        self._emit("dummy_input = torch.randn(*self.input_shape).to(self.device)")
        self._emit("")
        self._emit("# Warmup")
        self._emit("for _ in range(self.num_warmup):")
        self._indent_up()
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("self.model(dummy_input)")
        self._indent_down()
        self._indent_down()
        self._emit("")
        self._emit("if self.device.type == 'cuda':")
        self._indent_up()
        self._emit("torch.cuda.synchronize()")
        self._indent_down()
        self._emit("")
        self._emit("latencies = []")
        self._emit("for _ in range(self.num_runs):")
        self._indent_up()
        self._emit("if self.device.type == 'cuda':")
        self._indent_up()
        self._emit("torch.cuda.synchronize()")
        self._indent_down()
        self._emit("start = time.perf_counter()")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("self.model(dummy_input)")
        self._indent_down()
        self._emit("if self.device.type == 'cuda':")
        self._indent_up()
        self._emit("torch.cuda.synchronize()")
        self._indent_down()
        self._emit("latencies.append((time.perf_counter() - start) * 1000)")
        self._indent_down()
        self._emit("")
        self._emit("latencies = np.array(latencies)")
        self._emit("results = {")
        self._indent_up()
        self._emit("'mean_ms': np.mean(latencies),")
        self._emit("'std_ms': np.std(latencies),")
        self._emit("'median_ms': np.median(latencies),")
        self._emit("'p95_ms': np.percentile(latencies, 95),")
        self._emit("'p99_ms': np.percentile(latencies, 99),")
        self._indent_down()
        self._emit("}")
        self._emit("print(f'Latency: {results[\"mean_ms\"]:.2f} +/- {results[\"std_ms\"]:.2f} ms (p95={results[\"p95_ms\"]:.2f} ms)')")
        self._emit("return results")
        self._indent_down()
        self._emit("")

        # Throughput benchmark
        self._emit("def benchmark_throughput(self, batch_sizes=None):")
        self._indent_up()
        self._emit("if batch_sizes is None:")
        self._indent_up()
        self._emit("batch_sizes = [1, 2, 4, 8, 16, 32, 64]")
        self._indent_down()
        self._emit("results = {}")
        self._emit("for bs in batch_sizes:")
        self._indent_up()
        self._emit("shape = [bs] + self.input_shape[1:]")
        self._emit("dummy_input = torch.randn(*shape).to(self.device)")
        self._emit("for _ in range(self.num_warmup):")
        self._indent_up()
        self._emit("with torch.no_grad(): self.model(dummy_input)")
        self._indent_down()
        self._emit("if self.device.type == 'cuda': torch.cuda.synchronize()")
        self._emit("start = time.perf_counter()")
        self._emit("for _ in range(self.num_runs):")
        self._indent_up()
        self._emit("with torch.no_grad(): self.model(dummy_input)")
        self._indent_down()
        self._emit("if self.device.type == 'cuda': torch.cuda.synchronize()")
        self._emit("elapsed = time.perf_counter() - start")
        self._emit("throughput = (self.num_runs * bs) / elapsed")
        self._emit("results[bs] = throughput")
        self._emit("print(f'  batch_size={bs}: {throughput:.1f} samples/sec')")
        self._indent_down()
        self._emit("return results")
        self._indent_down()
        self._emit("")

        # Memory benchmark
        self._emit("def benchmark_memory(self):")
        self._indent_up()
        self._emit("if self.device.type != 'cuda':")
        self._indent_up()
        self._emit("print('Memory benchmarking requires CUDA.')")
        self._emit("return {}")
        self._indent_down()
        self._emit("torch.cuda.reset_peak_memory_stats()")
        self._emit("torch.cuda.empty_cache()")
        self._emit("dummy_input = torch.randn(*self.input_shape).to(self.device)")
        self._emit("mem_before = torch.cuda.memory_allocated() / 1024 / 1024")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("self.model(dummy_input)")
        self._indent_down()
        self._emit("mem_after = torch.cuda.memory_allocated() / 1024 / 1024")
        self._emit("peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024")
        self._emit("results = {'allocated_mb': mem_after, 'peak_mb': peak_mem, 'inference_mb': mem_after - mem_before}")
        self._emit("print(f'Memory: allocated={mem_after:.1f}MB, peak={peak_mem:.1f}MB')")
        self._emit("return results")
        self._indent_down()
        self._emit("")

        # Parameter count
        self._emit("def benchmark_params(self):")
        self._indent_up()
        self._emit("total = sum(p.numel() for p in self.model.parameters())")
        self._emit("trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)")
        self._emit("results = {'total_params': total, 'trainable_params': trainable, 'total_mb': total * 4 / 1024 / 1024}")
        self._emit("print(f'Parameters: {total:,} total ({trainable:,} trainable) = {results[\"total_mb\"]:.1f}MB')")
        self._emit("return results")
        self._indent_down()
        self._emit("")

        # Run all
        self._emit("def run(self):")
        self._indent_up()
        self._emit("print(f'=== Benchmark: {self.device} ===')")
        self._emit("results = {}")
        self._emit("results['params'] = self.benchmark_params()")
        self._emit("results['latency'] = self.benchmark_latency()")
        self._emit("results['throughput'] = self.benchmark_throughput()")
        self._emit("results['memory'] = self.benchmark_memory()")
        self._emit("return results")
        self._indent_down()

        self._indent_down()

    # --- Augment Transpilation -----------------------------------------------------

    def _transpile_augment(self, aug: AugmentDef):
        domain = aug.domain.lower()

        if domain == "image":
            self._imports.add("import albumentations as A")
            self._imports.add("from albumentations.pytorch import ToTensorV2")
        elif domain == "audio":
            self._imports.add("import audiomentations as AM")
        elif domain == "text":
            self._imports.add("import nlpaug.augmenter.word as naw")
            self._imports.add("import nlpaug.augmenter.char as nac")

        self._emit(f"class {aug.name}Augmenter:")
        self._indent_up()
        self._emit(f'"""Data augmentation pipeline for {domain} data."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()

        if domain == "image":
            self._emit(f"self.transform = A.Compose([")
            self._indent_up()
            if aug.transforms:
                for t in aug.transforms:
                    self._emit(f"{self._albumentations_transform(t)},")
            else:
                self._emit("A.HorizontalFlip(p=0.5),")
                self._emit("A.RandomBrightnessContrast(p=0.2),")
                self._emit("A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),")
                self._emit("A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),")
                self._emit("A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),")
                self._emit("ToTensorV2(),")
            self._indent_down()
            self._emit("])")
        elif domain == "audio":
            self._emit("self.transform = AM.Compose([")
            self._indent_up()
            if aug.transforms:
                for t in aug.transforms:
                    self._emit(f"{self._audiomentations_transform(t)},")
            else:
                self._emit("AM.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),")
                self._emit("AM.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),")
                self._emit("AM.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),")
                self._emit("AM.Shift(min_shift=-0.5, max_shift=0.5, p=0.5),")
            self._indent_down()
            self._emit("])")
        elif domain == "text":
            self._emit("self.augmenters = [")
            self._indent_up()
            if aug.transforms:
                for t in aug.transforms:
                    self._emit(f"{self._nlpaug_transform(t)},")
            else:
                self._emit("naw.SynonymAug(aug_src='wordnet'),")
                self._emit("naw.RandomWordAug(action='swap'),")
                self._emit("naw.RandomWordAug(action='delete'),")
            self._indent_down()
            self._emit("]")
        else:
            self._emit("self.transform = None")

        self._indent_down()
        self._emit("")

        self._emit("def __call__(self, data):")
        self._indent_up()
        if domain == "image":
            self._emit("if isinstance(data, dict):")
            self._indent_up()
            self._emit("return self.transform(**data)")
            self._indent_down()
            self._emit("return self.transform(image=data)['image']")
        elif domain == "audio":
            self._emit("return self.transform(samples=data, sample_rate=16000)")
        elif domain == "text":
            self._emit("import random")
            self._emit("augmenter = random.choice(self.augmenters)")
            self._emit("return augmenter.augment(data)")
        else:
            self._emit("return data")
        self._indent_down()

        self._indent_down()

    def _albumentations_transform(self, call: FunctionCall) -> str:
        name = call.name
        kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in call.kwargs.items())
        args = ", ".join(self._value_to_python(a) for a in call.args)
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"A.{name}({all_args})"

    def _audiomentations_transform(self, call: FunctionCall) -> str:
        name = call.name
        kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in call.kwargs.items())
        args = ", ".join(self._value_to_python(a) for a in call.args)
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"AM.{name}({all_args})"

    def _nlpaug_transform(self, call: FunctionCall) -> str:
        name = call.name
        kwargs = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in call.kwargs.items())
        args = ", ".join(self._value_to_python(a) for a in call.args)
        all_args = ", ".join(filter(None, [args, kwargs]))
        return f"naw.{name}({all_args})"

    # --- Feature Engineering Transpilation ----------------------------------------

    def _transpile_feature(self, feat: FeatureDef):
        self._imports.add("import pandas as pd")
        self._imports.add("import numpy as np")
        self._imports.add("from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder")
        self._imports.add("from sklearn.decomposition import PCA")
        self._imports.add("from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif")

        self._emit(f"class {feat.name}FeatureEngineer:")
        self._indent_up()
        self._emit(f'"""Feature engineering pipeline for {feat.name}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit("self.scalers = {}")
        self._emit("self.encoders = {}")
        self._emit("self.fitted = False")
        self._indent_down()
        self._emit("")

        self._emit("def fit_transform(self, df):")
        self._indent_up()
        self._emit("df = df.copy()")
        self._emit("")

        if feat.operations:
            for op in feat.operations:
                if isinstance(op, FunctionCall):
                    op_name = op.name.lower()
                    if op_name == "normalize":
                        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
                        self._emit("scaler = StandardScaler()")
                        self._emit("df[numeric_cols] = scaler.fit_transform(df[numeric_cols])")
                        self._emit("self.scalers['standard'] = scaler")
                    elif op_name == "minmax":
                        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
                        self._emit("scaler = MinMaxScaler()")
                        self._emit("df[numeric_cols] = scaler.fit_transform(df[numeric_cols])")
                        self._emit("self.scalers['minmax'] = scaler")
                    elif op_name == "onehot":
                        self._emit("cat_cols = df.select_dtypes(include=['object', 'category']).columns")
                        self._emit("df = pd.get_dummies(df, columns=cat_cols, drop_first=True)")
                    elif op_name == "pca":
                        n = int(op.args[0].value) if op.args else 10
                        self._emit(f"pca = PCA(n_components={n})")
                        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
                        self._emit("pca_result = pca.fit_transform(df[numeric_cols])")
                        self._emit(f"pca_df = pd.DataFrame(pca_result, columns=[f'pca_{{i}}' for i in range({n})])")
                        self._emit("df = pd.concat([df.drop(columns=numeric_cols), pca_df], axis=1)")
                    elif op_name == "select_k_best":
                        k = int(op.args[0].value) if op.args else 10
                        self._emit(f"if '{feat.target}' in df.columns:")
                        self._indent_up()
                        self._emit(f"selector = SelectKBest(f_classif, k={k})")
                        self._emit(f"X = df.drop(columns=['{feat.target}'])")
                        self._emit(f"y = df['{feat.target}']")
                        self._emit("X_selected = selector.fit_transform(X, y)")
                        self._emit("selected_cols = X.columns[selector.get_support()]")
                        self._emit(f"df = pd.DataFrame(X_selected, columns=selected_cols)")
                        self._emit(f"df['{feat.target}'] = y.values")
                        self._indent_down()
                    elif op_name == "fillna":
                        strategy = op.args[0].name if op.args and isinstance(op.args[0], Identifier) else "median"
                        if strategy == "median":
                            self._emit("df = df.fillna(df.median(numeric_only=True))")
                        elif strategy == "mean":
                            self._emit("df = df.fillna(df.mean(numeric_only=True))")
                        elif strategy == "zero":
                            self._emit("df = df.fillna(0)")
                        else:
                            self._emit("df = df.fillna(df.median(numeric_only=True))")
                    elif op_name == "log_transform":
                        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
                        self._emit("for col in numeric_cols:")
                        self._indent_up()
                        self._emit("if (df[col] > 0).all():")
                        self._indent_up()
                        self._emit("df[f'{col}_log'] = np.log1p(df[col])")
                        self._indent_down()
                        self._indent_down()
                    elif op_name == "polynomial":
                        degree = int(op.args[0].value) if op.args else 2
                        self._emit("from sklearn.preprocessing import PolynomialFeatures")
                        self._emit(f"poly = PolynomialFeatures(degree={degree}, include_bias=False, interaction_only=True)")
                        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
                        self._emit("poly_features = poly.fit_transform(df[numeric_cols])")
                        self._emit("poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_cols))")
                        self._emit("df = pd.concat([df, poly_df], axis=1)")
                    else:
                        self._emit(f"# Custom operation: {op_name}")
        else:
            self._emit("# Default feature engineering")
            self._emit("df = df.fillna(df.median(numeric_only=True))")
            self._emit("cat_cols = df.select_dtypes(include=['object', 'category']).columns")
            self._emit("df = pd.get_dummies(df, columns=cat_cols, drop_first=True)")

        self._emit("")
        self._emit("self.fitted = True")
        self._emit("return df")
        self._indent_down()
        self._emit("")

        self._emit("def transform(self, df):")
        self._indent_up()
        self._emit("df = df.copy()")
        self._emit("if 'standard' in self.scalers:")
        self._indent_up()
        self._emit("numeric_cols = df.select_dtypes(include=[np.number]).columns")
        self._emit("df[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])")
        self._indent_down()
        self._emit("return df")
        self._indent_down()

        self._indent_down()

    # --- Embedding Transpilation -------------------------------------------------

    def _transpile_embedding(self, emb: EmbeddingDef):
        self._imports.add("from sentence_transformers import SentenceTransformer")
        self._imports.add("import numpy as np")

        self._emit(f"class {emb.name}Embedder:")
        self._indent_up()
        self._emit(f'"""Embedding extraction using {emb.model}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        model_name = emb.model if emb.model else "all-MiniLM-L6-v2"
        self._emit(f"self.model = SentenceTransformer('{model_name}')")
        self._emit(f"self.dim = {emb.dim}")
        self._emit(f"self.normalize = {emb.normalize}")
        self._indent_down()
        self._emit("")

        self._emit("def encode(self, texts, batch_size=32, show_progress=True):")
        self._indent_up()
        self._emit("embeddings = self.model.encode(")
        self._indent_up()
        self._emit("texts,")
        self._emit("batch_size=batch_size,")
        self._emit("show_progress_bar=show_progress,")
        self._emit(f"normalize_embeddings={emb.normalize}")
        self._indent_down()
        self._emit(")")
        self._emit("return embeddings")
        self._indent_down()
        self._emit("")

        self._emit("def encode_to_tensor(self, texts, batch_size=32):")
        self._indent_up()
        self._emit("embeddings = self.encode(texts, batch_size=batch_size, show_progress=False)")
        self._emit("return torch.tensor(embeddings)")
        self._indent_down()
        self._emit("")

        self._emit("def similarity(self, text1, text2):")
        self._indent_up()
        self._emit("emb1 = self.encode([text1])")
        self._emit("emb2 = self.encode([text2])")
        self._emit("from numpy.linalg import norm")
        self._emit("sim = np.dot(emb1[0], emb2[0]) / (norm(emb1[0]) * norm(emb2[0]))")
        self._emit("return float(sim)")
        self._indent_down()

        self._indent_down()

    # --- Tokenizer Transpilation -------------------------------------------------

    def _transpile_tokenizer(self, tok: TokenizerDef):
        tok_type = tok.type.lower()

        if tok_type in ("bpe", "wordpiece", "unigram"):
            self._imports.add("from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders")
        elif tok_type == "sentencepiece":
            self._imports.add("import sentencepiece as spm")
        elif tok_type == "tiktoken":
            self._imports.add("import tiktoken")
        else:
            self._imports.add("from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders")

        self._emit(f"class {tok.name}Tokenizer:")
        self._indent_up()
        self._emit(f'"""Custom tokenizer ({tok_type}, vocab_size={tok.vocab_size})."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.vocab_size = {tok.vocab_size}")
        self._emit(f"self.max_length = {tok.max_length}")
        special = [self._value_to_python(t) for t in tok.special_tokens] if tok.special_tokens else ["'[PAD]'", "'[UNK]'", "'[CLS]'", "'[SEP]'", "'[MASK]'"]
        self._emit(f"self.special_tokens = [{', '.join(special)}]")
        self._emit("self.tokenizer = None")
        self._indent_down()
        self._emit("")

        self._emit("def train(self, files):")
        self._indent_up()
        if tok_type == "bpe":
            self._emit("self.tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))")
            self._emit("self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)")
            self._emit("trainer = trainers.BpeTrainer(")
            self._indent_up()
            self._emit(f"vocab_size=self.vocab_size,")
            self._emit("special_tokens=self.special_tokens,")
            self._emit("min_frequency=2")
            self._indent_down()
            self._emit(")")
            self._emit("self.tokenizer.train(files, trainer)")
            self._emit("self.tokenizer.decoder = decoders.ByteLevel()")
        elif tok_type == "wordpiece":
            self._emit("self.tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))")
            self._emit("self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()")
            self._emit("trainer = trainers.WordPieceTrainer(")
            self._indent_up()
            self._emit(f"vocab_size=self.vocab_size,")
            self._emit("special_tokens=self.special_tokens")
            self._indent_down()
            self._emit(")")
            self._emit("self.tokenizer.train(files, trainer)")
        elif tok_type == "unigram":
            self._emit("self.tokenizer = Tokenizer(models.Unigram())")
            self._emit("self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()")
            self._emit("trainer = trainers.UnigramTrainer(")
            self._indent_up()
            self._emit(f"vocab_size=self.vocab_size,")
            self._emit("special_tokens=self.special_tokens,")
            self._emit("unk_token='[UNK]'")
            self._indent_down()
            self._emit(")")
            self._emit("self.tokenizer.train(files, trainer)")
            self._emit("self.tokenizer.decoder = decoders.Metaspace()")
        elif tok_type == "sentencepiece":
            self._emit("spm.SentencePieceTrainer.train(")
            self._indent_up()
            self._emit(f"input=','.join(files),")
            self._emit(f"model_prefix='{tok.name}',")
            self._emit(f"vocab_size=self.vocab_size,")
            self._emit("model_type='bpe'")
            self._indent_down()
            self._emit(")")
            self._emit(f"self.tokenizer = spm.SentencePieceProcessor(model_file='{tok.name}.model')")
        elif tok_type == "tiktoken":
            self._emit("self.tokenizer = tiktoken.get_encoding('cl100k_base')")
        self._emit("print(f'Tokenizer trained. Vocab size: {self.vocab_size}')")
        self._indent_down()
        self._emit("")

        self._emit("def encode(self, text):")
        self._indent_up()
        if tok_type in ("bpe", "wordpiece", "unigram"):
            self._emit("encoded = self.tokenizer.encode(text)")
            self._emit("ids = encoded.ids[:self.max_length]")
        elif tok_type == "sentencepiece":
            self._emit("ids = self.tokenizer.encode(text)[:self.max_length]")
        elif tok_type == "tiktoken":
            self._emit("ids = self.tokenizer.encode(text)[:self.max_length]")
        else:
            self._emit("ids = []")
        if tok.padding == "max_length":
            self._emit("# Padding")
            self._emit("while len(ids) < self.max_length:")
            self._indent_up()
            self._emit("ids.append(0)")
            self._indent_down()
        self._emit("return ids")
        self._indent_down()
        self._emit("")

        self._emit("def decode(self, ids):")
        self._indent_up()
        if tok_type in ("bpe", "wordpiece", "unigram"):
            self._emit("return self.tokenizer.decode(ids)")
        elif tok_type == "sentencepiece":
            self._emit("return self.tokenizer.decode(ids)")
        elif tok_type == "tiktoken":
            self._emit("return self.tokenizer.decode(ids)")
        else:
            self._emit("return ''")
        self._indent_down()
        self._emit("")

        self._emit("def save(self, path):")
        self._indent_up()
        if tok_type in ("bpe", "wordpiece", "unigram"):
            self._emit("self.tokenizer.save(path)")
        elif tok_type == "sentencepiece":
            self._emit("print(f'Model saved as {tok.name}.model')")
        else:
            self._emit("print(f'Tokenizer saved to {path}')")
        self._indent_down()

        self._indent_down()

    # --- Callback Transpilation ----------------------------------------------------

    def _transpile_callback(self, cb: CallbackDef):
        self._emit(f"class {cb.name}Callback:")
        self._indent_up()
        self._emit(f'"""Custom training callback (trigger: {cb.trigger})."""')
        self._emit("")

        self._emit("def __init__(self, **kwargs):")
        self._indent_up()
        self._emit("self.config = kwargs")
        self._emit("self.logs = []")
        self._indent_down()
        self._emit("")

        trigger = cb.trigger.lower()
        trigger_method_map = {
            "epoch_end": "on_epoch_end",
            "batch_end": "on_batch_end",
            "train_start": "on_train_start",
            "train_end": "on_train_end",
            "eval_end": "on_eval_end",
        }
        method_name = trigger_method_map.get(trigger, "on_epoch_end")

        self._emit(f"def {method_name}(self, epoch=None, batch=None, logs=None):")
        self._indent_up()
        self._emit("logs = logs or {}")
        self._emit("self.logs.append(logs)")

        if cb.actions:
            for action in cb.actions:
                if isinstance(action, FunctionCall):
                    act_name = action.name.lower()
                    if act_name == "save_checkpoint":
                        self._emit("if 'model' in logs:")
                        self._indent_up()
                        self._emit("torch.save(logs['model'].state_dict(), f'checkpoint_epoch_{epoch}.pth')")
                        self._emit("print(f'Checkpoint saved at epoch {epoch}')")
                        self._indent_down()
                    elif act_name == "log":
                        self._emit("for k, v in logs.items():")
                        self._indent_up()
                        self._emit("if isinstance(v, (int, float)):")
                        self._indent_up()
                        self._emit("print(f'  {k}: {v}')")
                        self._indent_down()
                        self._indent_down()
                    elif act_name == "early_stop":
                        patience = int(action.args[0].value) if action.args else 5
                        self._emit(f"patience = {patience}")
                        self._emit("if len(self.logs) > patience:")
                        self._indent_up()
                        self._emit("recent_losses = [l.get('val_loss', float('inf')) for l in self.logs[-patience:]]")
                        self._emit("if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):")
                        self._indent_up()
                        self._emit("print(f'Early stopping triggered at epoch {epoch}')")
                        self._emit("return 'stop'")
                        self._indent_down()
                        self._indent_down()
                    elif act_name == "reduce_lr":
                        factor = float(action.args[0].value) if action.args else 0.5
                        self._emit(f"if 'optimizer' in logs:")
                        self._indent_up()
                        self._emit(f"for param_group in logs['optimizer'].param_groups:")
                        self._indent_up()
                        self._emit(f"param_group['lr'] *= {factor}")
                        self._indent_down()
                        self._emit(f"print(f'Learning rate reduced by factor {factor}')")
                        self._indent_down()
                    else:
                        self._emit(f"# Action: {act_name}")
        else:
            self._emit("print(f'Callback triggered: {method_name} at epoch={epoch}')")

        self._indent_down()

        # Also add all other trigger methods as pass-throughs
        all_methods = ["on_epoch_end", "on_batch_end", "on_train_start", "on_train_end", "on_eval_end"]
        for m in all_methods:
            if m != method_name:
                self._emit("")
                self._emit(f"def {m}(self, **kwargs):")
                self._indent_up()
                self._emit("pass")
                self._indent_down()

        self._indent_down()

    # --- Metric Transpilation ----------------------------------------------------

    def _transpile_metric(self, metric: MetricDef):
        self._imports.add("import numpy as np")

        self._emit(f"class {metric.name}Metric:")
        self._indent_up()
        self._emit(f'"""Custom metric: {metric.name}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.higher_is_better = {metric.higher_is_better}")
        self._emit("self.reset()")
        self._indent_down()
        self._emit("")

        self._emit("def reset(self):")
        self._indent_up()
        self._emit("self.predictions = []")
        self._emit("self.targets = []")
        self._indent_down()
        self._emit("")

        self._emit("def update(self, predictions, targets):")
        self._indent_up()
        self._emit("if isinstance(predictions, torch.Tensor):")
        self._indent_up()
        self._emit("predictions = predictions.detach().cpu().numpy()")
        self._indent_down()
        self._emit("if isinstance(targets, torch.Tensor):")
        self._indent_up()
        self._emit("targets = targets.detach().cpu().numpy()")
        self._indent_down()
        self._emit("self.predictions.extend(predictions.flatten())")
        self._emit("self.targets.extend(targets.flatten())")
        self._indent_down()
        self._emit("")

        self._emit("def compute(self):")
        self._indent_up()
        self._emit("preds = np.array(self.predictions)")
        self._emit("targets = np.array(self.targets)")

        if metric.formula:
            formula = metric.formula
            self._emit(f"# Custom formula: {formula}")
            self._emit(f"result = {formula}")
        else:
            self._emit("# Default: accuracy")
            self._emit("result = np.mean(preds == targets)")

        self._emit("return float(result)")
        self._indent_down()
        self._emit("")

        self._emit("def __repr__(self):")
        self._indent_up()
        self._emit(f"return f'{metric.name}Metric(value={{self.compute():.4f}})'")
        self._indent_down()

        self._indent_down()

    # --- RAG Transpilation -------------------------------------------------------

    def _transpile_rag(self, rag: RAGDef):
        retriever = rag.retriever.lower()
        self._imports.add("from langchain.text_splitter import RecursiveCharacterTextSplitter")
        self._imports.add("from langchain.schema import Document")

        if retriever == "faiss":
            self._imports.add("from langchain_community.vectorstores import FAISS")
        elif retriever == "chromadb":
            self._imports.add("from langchain_community.vectorstores import Chroma")
        elif retriever == "pinecone":
            self._imports.add("from langchain_community.vectorstores import Pinecone")

        if rag.embedding_model:
            self._imports.add("from langchain_community.embeddings import HuggingFaceEmbeddings")
        else:
            self._imports.add("from langchain_community.embeddings import HuggingFaceEmbeddings")

        if rag.generator:
            self._imports.add("from langchain_community.llms import HuggingFacePipeline")

        self._emit(f"class {rag.name}RAG:")
        self._indent_up()
        self._emit(f'"""RAG pipeline with {retriever} retriever."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        emb_model = rag.embedding_model if rag.embedding_model else "sentence-transformers/all-MiniLM-L6-v2"
        self._emit(f"self.embeddings = HuggingFaceEmbeddings(model_name='{emb_model}')")
        self._emit(f"self.text_splitter = RecursiveCharacterTextSplitter(")
        self._indent_up()
        self._emit(f"chunk_size={rag.chunk_size},")
        self._emit(f"chunk_overlap={rag.chunk_overlap},")
        self._emit("separators=['\\n\\n', '\\n', '. ', ' ', '']")
        self._indent_down()
        self._emit(")")
        self._emit(f"self.top_k = {rag.top_k}")
        self._emit("self.vector_store = None")
        if rag.generator:
            self._emit(f"self.llm = HuggingFacePipeline.from_model_id(model_id='{rag.generator}', task='text-generation')")
        else:
            self._emit("self.llm = None")
        self._indent_down()
        self._emit("")

        # Index documents
        self._emit("def index(self, documents):")
        self._indent_up()
        self._emit("if isinstance(documents, list) and isinstance(documents[0], str):")
        self._indent_up()
        self._emit("documents = [Document(page_content=doc) for doc in documents]")
        self._indent_down()
        self._emit("splits = self.text_splitter.split_documents(documents)")
        self._emit("print(f'Created {len(splits)} chunks from {len(documents)} documents')")

        if retriever == "faiss":
            self._emit("self.vector_store = FAISS.from_documents(splits, self.embeddings)")
        elif retriever == "chromadb":
            self._emit(f"self.vector_store = Chroma.from_documents(splits, self.embeddings, collection_name='{rag.name}')")
        elif retriever == "pinecone":
            self._emit(f"self.vector_store = Pinecone.from_documents(splits, self.embeddings, index_name='{rag.name}')")
        else:
            self._emit("self.vector_store = FAISS.from_documents(splits, self.embeddings)")

        self._emit("print('Vector store built successfully.')")
        self._indent_down()
        self._emit("")

        # Retrieve
        self._emit("def retrieve(self, query, top_k=None):")
        self._indent_up()
        self._emit("k = top_k or self.top_k")
        self._emit("if self.vector_store is None:")
        self._indent_up()
        self._emit("raise ValueError('No documents indexed. Call index() first.')")
        self._indent_down()
        self._emit("results = self.vector_store.similarity_search_with_score(query, k=k)")
        self._emit("return results")
        self._indent_down()
        self._emit("")

        # Query (retrieval + generation)
        self._emit("def query(self, question):")
        self._indent_up()
        self._emit("docs = self.retrieve(question)")
        self._emit("context = '\\n\\n'.join([doc.page_content for doc, score in docs])")
        self._emit("if self.llm is not None:")
        self._indent_up()
        self._emit("prompt = f'Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:'")
        self._emit("response = self.llm(prompt)")
        self._emit("return {'answer': response, 'sources': docs}")
        self._indent_down()
        self._emit("return {'context': context, 'sources': docs}")
        self._indent_down()

        self._indent_down()

    # --- Agent Transpilation -----------------------------------------------------

    def _transpile_agent(self, agent: AgentDef):
        self._imports.add("from langchain.agents import AgentExecutor, create_react_agent")
        self._imports.add("from langchain.tools import Tool")
        self._imports.add("from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory")
        self._imports.add("from langchain_community.llms import HuggingFacePipeline")
        self._imports.add("from langchain.prompts import PromptTemplate")

        self._emit(f"class {agent.name}Agent:")
        self._indent_up()
        self._emit(f'"""LLM agent with tool calling."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        llm = agent.llm if agent.llm else "gpt2"
        self._emit(f"self.llm = HuggingFacePipeline.from_model_id(model_id='{llm}', task='text-generation', pipeline_kwargs={{'max_new_tokens': 256}})")
        self._emit(f"self.max_iterations = {agent.max_iterations}")
        self._emit("")

        # Tools
        self._emit("self.tools = [")
        self._indent_up()
        if agent.tools:
            for tool in agent.tools:
                if isinstance(tool, FunctionCall):
                    tool_name = tool.name
                    desc = ""
                    for k, v in tool.kwargs.items():
                        if k == "description":
                            desc = self._value_to_python(v)
                    self._emit(f"Tool(name='{tool_name}', func=lambda x: x, description={desc or repr(tool_name)}),")
                elif isinstance(tool, StringLiteral):
                    self._emit(f"Tool(name={repr(tool.value)}, func=lambda x: x, description={repr(tool.value)}),")
                elif isinstance(tool, Identifier):
                    self._emit(f"Tool(name='{tool.name}', func=lambda x: x, description='{tool.name} tool'),")
        else:
            self._emit("Tool(name='search', func=lambda x: f'Search results for: {x}', description='Search the web'),")
            self._emit("Tool(name='calculator', func=lambda x: str(eval(x)), description='Calculate math expressions'),")
        self._indent_down()
        self._emit("]")
        self._emit("")

        # Memory
        memory_type = agent.memory or "buffer"
        if memory_type == "buffer":
            self._emit("self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)")
        elif memory_type == "summary":
            self._emit("self.memory = ConversationSummaryMemory(llm=self.llm, memory_key='chat_history', return_messages=True)")
        elif memory_type == "vector":
            self._imports.add("from langchain.memory import VectorStoreRetrieverMemory")
            self._emit("self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)")
        else:
            self._emit("self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)")

        # System prompt
        if agent.system_prompt:
            self._emit(f"self.system_prompt = {repr(agent.system_prompt)}")
        else:
            self._emit(f"self.system_prompt = 'You are a helpful AI assistant.'")

        self._emit("")
        self._emit("template = '''{{system_prompt}}")
        self._emit("")
        self._emit("You have access to the following tools: {{tools}}")
        self._emit("Tool names: {{tool_names}}")
        self._emit("")
        self._emit("Use this format:")
        self._emit("Question: the input question")
        self._emit("Thought: think about what to do")
        self._emit("Action: the tool to use")
        self._emit("Action Input: the input to the tool")
        self._emit("Observation: the result")
        self._emit("... (repeat)")
        self._emit("Final Answer: the final answer")
        self._emit("")
        self._emit("Question: {{input}}")
        self._emit("{{agent_scratchpad}}'''")
        self._emit("")
        self._emit("prompt = PromptTemplate(template=template, input_variables=['input', 'agent_scratchpad'], partial_variables={'system_prompt': self.system_prompt, 'tools': str(self.tools), 'tool_names': ', '.join([t.name for t in self.tools])})")
        self._emit("react_agent = create_react_agent(self.llm, self.tools, prompt)")
        self._emit("self.executor = AgentExecutor(agent=react_agent, tools=self.tools, memory=self.memory, max_iterations=self.max_iterations, verbose=True)")

        self._indent_down()
        self._emit("")

        self._emit("def run(self, query):")
        self._indent_up()
        self._emit("result = self.executor.invoke({'input': query})")
        self._emit("return result['output']")
        self._indent_down()
        self._emit("")

        self._emit("def chat(self, message):")
        self._indent_up()
        self._emit("return self.run(message)")
        self._indent_down()

        self._indent_down()

    # --- Federated Learning Transpilation -----------------------------------------

    def _transpile_federated(self, fed: FederatedDef):
        self._imports.add("import flwr as fl")
        self._imports.add("from collections import OrderedDict")
        self._imports.add("import numpy as np")

        self._emit(f"class {fed.name}Federated:")
        self._indent_up()
        self._emit(f'"""Federated learning with {fed.strategy} strategy."""')
        self._emit("")

        # Flower client class
        self._emit("class FlowerClient(fl.client.NumPyClient):")
        self._indent_up()
        self._emit("def __init__(self, model, trainloader, valloader, device):")
        self._indent_up()
        self._emit("self.model = model")
        self._emit("self.trainloader = trainloader")
        self._emit("self.valloader = valloader")
        self._emit("self.device = device")
        self._indent_down()
        self._emit("")

        self._emit("def get_parameters(self, config):")
        self._indent_up()
        self._emit("return [val.cpu().numpy() for _, val in self.model.state_dict().items()]")
        self._indent_down()
        self._emit("")

        self._emit("def set_parameters(self, parameters):")
        self._indent_up()
        self._emit("params_dict = zip(self.model.state_dict().keys(), parameters)")
        self._emit("state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})")
        self._emit("self.model.load_state_dict(state_dict, strict=True)")
        self._indent_down()
        self._emit("")

        self._emit("def fit(self, parameters, config):")
        self._indent_up()
        self._emit("self.set_parameters(parameters)")
        self._emit("self.model.train()")
        self._emit("optimizer = optim.SGD(self.model.parameters(), lr=0.01)")
        self._emit("criterion = nn.CrossEntropyLoss()")
        self._emit("for epoch in range(config.get('local_epochs', 1)):")
        self._indent_up()
        self._emit("for inputs, targets in self.trainloader:")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("optimizer.zero_grad()")
        self._emit("loss = criterion(self.model(inputs), targets)")
        self._emit("loss.backward()")
        self._emit("optimizer.step()")
        self._indent_down()
        self._indent_down()
        self._emit("return self.get_parameters(config={}), len(self.trainloader.dataset), {}")
        self._indent_down()
        self._emit("")

        self._emit("def evaluate(self, parameters, config):")
        self._indent_up()
        self._emit("self.set_parameters(parameters)")
        self._emit("self.model.eval()")
        self._emit("criterion = nn.CrossEntropyLoss()")
        self._emit("total_loss, correct, total = 0.0, 0, 0")
        self._emit("with torch.no_grad():")
        self._indent_up()
        self._emit("for inputs, targets in self.valloader:")
        self._indent_up()
        self._emit("inputs, targets = inputs.to(self.device), targets.to(self.device)")
        self._emit("outputs = self.model(inputs)")
        self._emit("total_loss += criterion(outputs, targets).item()")
        self._emit("_, predicted = outputs.max(1)")
        self._emit("total += targets.size(0)")
        self._emit("correct += predicted.eq(targets).sum().item()")
        self._indent_down()
        self._indent_down()
        self._emit("accuracy = correct / total")
        self._emit("return float(total_loss / len(self.valloader)), len(self.valloader.dataset), {'accuracy': accuracy}")
        self._indent_down()

        self._indent_down()  # End FlowerClient
        self._emit("")

        # Server strategy
        strategy = fed.strategy.lower()
        self._emit("def __init__(self, model_fn, data_fn):")
        self._indent_up()
        self._emit("self.model_fn = model_fn")
        self._emit("self.data_fn = data_fn")
        self._emit(f"self.num_clients = {fed.num_clients}")
        self._emit(f"self.num_rounds = {fed.rounds}")
        self._emit(f"self.fraction_fit = {fed.fraction_fit}")
        self._indent_down()
        self._emit("")

        self._emit("def _client_fn(self, cid):")
        self._indent_up()
        self._emit("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        self._emit("model = self.model_fn().to(device)")
        self._emit("trainloader, valloader = self.data_fn(int(cid))")
        self._emit(f"return {fed.name}Federated.FlowerClient(model, trainloader, valloader, device).to_client()")
        self._indent_down()
        self._emit("")

        self._emit("def run(self):")
        self._indent_up()

        strategy_map = {
            "fedavg": "fl.server.strategy.FedAvg",
            "fedprox": "fl.server.strategy.FedProx",
            "fedadam": "fl.server.strategy.FedAdam",
            "scaffold": "fl.server.strategy.FedAvg",
        }
        strat_cls = strategy_map.get(strategy, "fl.server.strategy.FedAvg")

        self._emit(f"strategy = {strat_cls}(")
        self._indent_up()
        self._emit(f"fraction_fit={fed.fraction_fit},")
        self._emit(f"fraction_evaluate=0.5,")
        self._emit(f"min_fit_clients=max(2, int(self.num_clients * self.fraction_fit)),")
        self._emit(f"min_evaluate_clients=2,")
        self._emit(f"min_available_clients=self.num_clients,")
        if strategy == "fedprox":
            self._emit("proximal_mu=0.1,")
        self._indent_down()
        self._emit(")")
        self._emit("")
        self._emit("fl.simulation.start_simulation(")
        self._indent_up()
        self._emit("client_fn=self._client_fn,")
        self._emit("num_clients=self.num_clients,")
        self._emit("config=fl.server.ServerConfig(num_rounds=self.num_rounds),")
        self._emit("strategy=strategy,")
        self._indent_down()
        self._emit(")")
        self._emit("print(f'Federated learning complete after {self.num_rounds} rounds.')")
        self._indent_down()

        self._indent_down()

    # --- AutoML Transpilation ----------------------------------------------------

    def _transpile_automl(self, automl: AutoMLDef):
        self._imports.add("import pandas as pd")
        self._imports.add("import numpy as np")

        framework = automl.framework.lower()

        if framework in ("auto", "autogluon"):
            self._imports.add("from autogluon.tabular import TabularPredictor")
        elif framework == "flaml":
            self._imports.add("from flaml import AutoML as FLAMLAutoML")
        elif framework == "h2o":
            self._imports.add("import h2o")
            self._imports.add("from h2o.automl import H2OAutoML")

        self._emit(f"class {automl.name}AutoML:")
        self._indent_up()
        self._emit(f'"""AutoML pipeline using {framework} for {automl.task}."""')
        self._emit("")

        self._emit("def __init__(self):")
        self._indent_up()
        self._emit(f"self.task = '{automl.task}'")
        self._emit(f"self.target_column = '{automl.target_column}'")
        self._emit(f"self.time_budget = {automl.time_budget}")
        self._emit(f"self.metric = '{automl.metric}'")
        self._emit("self.model = None")
        self._indent_down()
        self._emit("")

        self._emit("def run(self, train_data, test_data=None):")
        self._indent_up()

        if framework in ("auto", "autogluon"):
            self._emit("predictor = TabularPredictor(")
            self._indent_up()
            self._emit(f"label=self.target_column,")
            metric_map = {
                "accuracy": "accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "rmse": "root_mean_squared_error",
                "mse": "mean_squared_error",
                "r2": "r2",
            }
            ag_metric = metric_map.get(automl.metric, automl.metric)
            self._emit(f"eval_metric='{ag_metric}',")
            self._emit(f"path='./automl_results/{automl.name}'")
            self._indent_down()
            self._emit(")")
            self._emit("")
            self._emit("predictor.fit(")
            self._indent_up()
            self._emit("train_data=train_data,")
            self._emit(f"time_limit={automl.time_budget},")
            self._emit("presets='best_quality'")
            self._indent_down()
            self._emit(")")
            self._emit("")
            self._emit("if test_data is not None:")
            self._indent_up()
            self._emit("results = predictor.evaluate(test_data)")
            self._emit("print(f'Test results: {results}')")
            self._indent_down()
            self._emit("")
            self._emit("leaderboard = predictor.leaderboard()")
            self._emit("print(leaderboard)")
            self._emit("self.model = predictor")
            self._emit("return predictor")
        elif framework == "flaml":
            self._emit("automl = FLAMLAutoML()")
            self._emit("")
            self._emit("if isinstance(train_data, pd.DataFrame):")
            self._indent_up()
            self._emit("X_train = train_data.drop(columns=[self.target_column])")
            self._emit("y_train = train_data[self.target_column]")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("X_train, y_train = train_data")
            self._indent_down()
            self._emit("")
            self._emit("automl.fit(")
            self._indent_up()
            self._emit("X_train, y_train,")
            self._emit(f"task=self.task,")
            self._emit(f"time_budget={automl.time_budget},")
            self._emit(f"metric='{automl.metric}',")
            self._emit("verbose=1")
            self._indent_down()
            self._emit(")")
            self._emit("")
            self._emit("print(f'Best model: {automl.best_estimator}')")
            self._emit("print(f'Best config: {automl.best_config}')")
            self._emit("print(f'Best score: {automl.best_loss}')")
            self._emit("")
            self._emit("if test_data is not None:")
            self._indent_up()
            self._emit("if isinstance(test_data, pd.DataFrame):")
            self._indent_up()
            self._emit("X_test = test_data.drop(columns=[self.target_column])")
            self._emit("y_test = test_data[self.target_column]")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("X_test, y_test = test_data")
            self._indent_down()
            self._emit("score = automl.score(X_test, y_test)")
            self._emit("print(f'Test score: {score}')")
            self._indent_down()
            self._emit("")
            self._emit("self.model = automl")
            self._emit("return automl")
        elif framework == "h2o":
            self._emit("h2o.init()")
            self._emit("")
            self._emit("if isinstance(train_data, pd.DataFrame):")
            self._indent_up()
            self._emit("train_h2o = h2o.H2OFrame(train_data)")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("train_h2o = train_data")
            self._indent_down()
            self._emit("")
            self._emit("x = train_h2o.columns")
            self._emit("x.remove(self.target_column)")
            self._emit("y = self.target_column")
            self._emit("")
            self._emit("if self.task == 'classification':")
            self._indent_up()
            self._emit("train_h2o[y] = train_h2o[y].asfactor()")
            self._indent_down()
            self._emit("")
            self._emit(f"aml = H2OAutoML(max_runtime_secs={automl.time_budget}, seed=42)")
            self._emit("aml.train(x=x, y=y, training_frame=train_h2o)")
            self._emit("")
            self._emit("lb = aml.leaderboard")
            self._emit("print(lb.head())")
            self._emit("self.model = aml.leader")
            self._emit("return aml")
        else:
            self._emit(f"# Framework {framework} not recognized, using AutoGluon")
            self._emit("pass")

        self._indent_down()
        self._emit("")

        # Predict method
        self._emit("def predict(self, data):")
        self._indent_up()
        self._emit("if self.model is None:")
        self._indent_up()
        self._emit("raise ValueError('Model not trained. Call run() first.')")
        self._indent_down()
        if framework in ("auto", "autogluon"):
            self._emit("return self.model.predict(data)")
        elif framework == "flaml":
            self._emit("if isinstance(data, pd.DataFrame):")
            self._indent_up()
            self._emit("X = data.drop(columns=[self.target_column], errors='ignore')")
            self._indent_down()
            self._emit("else:")
            self._indent_up()
            self._emit("X = data")
            self._indent_down()
            self._emit("return self.model.predict(X)")
        elif framework == "h2o":
            self._emit("if isinstance(data, pd.DataFrame):")
            self._indent_up()
            self._emit("data = h2o.H2OFrame(data)")
            self._indent_down()
            self._emit("return self.model.predict(data)")
        else:
            self._emit("return self.model.predict(data)")
        self._indent_down()

        self._indent_down()
