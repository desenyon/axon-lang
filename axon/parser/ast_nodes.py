"""
Axon AST Node Definitions v2.0
================================
Comprehensive Abstract Syntax Tree for the Axon ML language.
Covers every ML domain: vision, NLP, audio, RL, GANs, diffusion,
graph neural nets, time series, tabular, multimodal, and more.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ═══════════════════════════════════════════════════════════════
# BASE NODES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """Base AST node."""
    line: int = 0
    col: int = 0


# ═══════════════════════════════════════════════════════════════
# TYPE SYSTEM
# ═══════════════════════════════════════════════════════════════

@dataclass
class TensorType(ASTNode):
    """Tensor[B, 3, 224, 224] shape annotation."""
    dims: list = field(default_factory=list)
    dtype: Optional[str] = None

@dataclass
class DataType(ASTNode):
    """Generic type annotation: List[int], Dict[str, Tensor], etc."""
    name: str = ""
    params: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# EXPRESSIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Identifier(ASTNode):
    name: str = ""

@dataclass
class NumberLiteral(ASTNode):
    value: Union[float, int] = 0

@dataclass
class StringLiteral(ASTNode):
    value: str = ""

@dataclass
class BoolLiteral(ASTNode):
    value: bool = False

@dataclass
class ListLiteral(ASTNode):
    elements: list = field(default_factory=list)

@dataclass
class DictLiteral(ASTNode):
    pairs: list = field(default_factory=list)  # list of (key, value) tuples

@dataclass
class FunctionCall(ASTNode):
    name: str = ""
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

@dataclass
class AttributeAccess(ASTNode):
    obj: ASTNode = field(default_factory=ASTNode)
    attr: str = ""

@dataclass
class ParamRef(ASTNode):
    """@param_name reference to configurable parameter."""
    name: str = ""

@dataclass
class ArrowExpr(ASTNode):
    """in_features -> out_features for Linear layers."""
    left: ASTNode = field(default_factory=ASTNode)
    right: ASTNode = field(default_factory=ASTNode)

@dataclass
class RatioExpr(ASTNode):
    """80/10/10 split ratios."""
    parts: list = field(default_factory=list)

@dataclass
class RangeExpr(ASTNode):
    """start..end for ranges."""
    start: Any = None
    end: Any = None
    step: Any = None

@dataclass
class Assignment(ASTNode):
    target: str = ""
    value: ASTNode = field(default_factory=ASTNode)

@dataclass
class KeyValue(ASTNode):
    """key: value pair within a block."""
    key: str = ""
    value: Any = None

@dataclass
class ConditionalExpr(ASTNode):
    """condition ? then : else"""
    condition: Any = None
    then_val: Any = None
    else_val: Any = None


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelDef(ASTNode):
    name: str = ""
    parent: Optional[str] = None
    layers: dict = field(default_factory=dict)
    forward_def: Optional['ForwardDef'] = None
    config: dict = field(default_factory=dict)

@dataclass
class ForwardDef(ASTNode):
    params: list = field(default_factory=list)
    return_type: Optional[TensorType] = None
    body: list = field(default_factory=list)

@dataclass
class LayerDef(ASTNode):
    layer_type: str = ""
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataDef(ASTNode):
    name: str = ""
    source: Any = None
    format: Optional[str] = None
    split: Optional[RatioExpr] = None
    transforms: list = field(default_factory=list)
    augmentations: list = field(default_factory=list)
    loader_config: dict = field(default_factory=dict)
    preprocessing: list = field(default_factory=list)
    streaming: bool = False
    cache: bool = False
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainDef(ASTNode):
    name: str = ""
    model_ref: Any = None
    data_ref: Any = None
    optimizer: Any = None
    scheduler: Any = None
    loss: Any = None
    epochs: int = 0
    device: str = "auto"
    precision: str = "fp32"
    callbacks: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    # Advanced training options
    gradient_accumulation: int = 1
    gradient_clip: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    ema: bool = False
    ema_decay: float = 0.999
    swa: bool = False
    swa_start: int = 0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    compile_model: bool = False
    # Distributed training
    distributed: Optional[str] = None  # ddp | fsdp | deepspeed
    num_gpus: int = 1
    num_nodes: int = 1
    # Logging
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 1
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalDef(ASTNode):
    name: str = ""
    checkpoint: str = "best"
    data_ref: Any = None
    metrics: list = field(default_factory=list)
    export: Optional[str] = None
    per_class: bool = False
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchDef(ASTNode):
    name: str = ""
    base_ref: str = ""
    method: str = "bayesian"
    trials: int = 50
    space: dict = field(default_factory=dict)
    objective: Any = None
    pruner: Any = None
    timeout: Optional[int] = None
    parallel_trials: int = 1
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class DeployDef(ASTNode):
    name: str = ""
    checkpoint: str = "best"
    format: str = "onnx"
    optimize: bool = False
    quantize: Optional[str] = None
    prune: Optional[str] = None
    serve_config: dict = field(default_factory=dict)
    docker: bool = False
    target_device: Optional[str] = None  # cpu | gpu | edge | mobile
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# FINE-TUNING
# ═══════════════════════════════════════════════════════════════

@dataclass
class FinetuneDef(ASTNode):
    name: str = ""
    base_model: str = ""
    method: str = "full"
    lora_config: dict = field(default_factory=dict)
    train_config: Optional[TrainDef] = None
    freeze_layers: Optional[list] = None
    unfreeze_layers: Optional[list] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineDef(ASTNode):
    name: str = ""
    steps: list = field(default_factory=list)
    parallel: bool = False
    config: dict = field(default_factory=dict)

@dataclass
class PipelineStep(ASTNode):
    name: str = ""
    block_ref: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleDef(ASTNode):
    name: str = ""
    models: list = field(default_factory=list)
    strategy: str = "voting"
    weights: Optional[list] = None
    meta_learner: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# EXPLAIN (INTERPRETABILITY)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExplainDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    method: str = "shap"
    data_ref: Optional[str] = None
    num_samples: int = 100
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# PRETRAIN
# ═══════════════════════════════════════════════════════════════

@dataclass
class PretrainDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    objective: str = "masked_lm"
    data_ref: Optional[str] = None
    train_config: Optional[TrainDef] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# GAN (Generative Adversarial Network)
# ═══════════════════════════════════════════════════════════════

@dataclass
class GANDef(ASTNode):
    name: str = ""
    generator: Any = None
    discriminator: Any = None
    latent_dim: int = 100
    loss_type: str = "vanilla"  # vanilla | wgan | wgan_gp | lsgan | hinge
    optimizer_g: Any = None
    optimizer_d: Any = None
    n_critic: int = 1
    gp_weight: float = 10.0
    epochs: int = 100
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DIFFUSION MODEL
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiffusionDef(ASTNode):
    name: str = ""
    model_ref: Any = None
    noise_scheduler: str = "ddpm"  # ddpm | ddim | euler | dpm
    timesteps: int = 1000
    beta_schedule: str = "linear"  # linear | cosine | sigmoid
    image_size: int = 256
    channels: int = 3
    guidance_scale: float = 7.5
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════════════════════

@dataclass
class RLDef(ASTNode):
    name: str = ""
    algorithm: str = "ppo"  # ppo | dqn | a2c | sac | td3 | ddpg | reinforce
    environment: str = ""
    policy: str = "MlpPolicy"
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    buffer_size: int = 100000
    reward_shaping: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TABULAR ML
# ═══════════════════════════════════════════════════════════════

@dataclass
class TabularDef(ASTNode):
    name: str = ""
    task: str = "classification"  # classification | regression | ranking
    algorithm: str = "xgboost"  # xgboost | lightgbm | catboost | random_forest | svm | knn | logistic | linear
    data_ref: Optional[str] = None
    target_column: str = ""
    feature_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    numerical_columns: list = field(default_factory=list)
    cross_validation: int = 5
    feature_engineering: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TIME SERIES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TimeSeriesDef(ASTNode):
    name: str = ""
    task: str = "forecast"  # forecast | anomaly | classification
    algorithm: str = "transformer"  # transformer | lstm | prophet | arima | nbeats | temporal_fusion
    data_ref: Optional[str] = None
    target_column: str = ""
    time_column: str = ""
    horizon: int = 24
    lookback: int = 168
    frequency: str = "1h"
    features: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# GRAPH NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════

@dataclass
class GraphDef(ASTNode):
    name: str = ""
    task: str = "node_classification"  # node_classification | graph_classification | link_prediction | graph_generation
    conv_type: str = "GCN"  # GCN | GAT | GraphSAGE | GIN | EdgeConv | MPNN
    layers: list = field(default_factory=list)
    num_features: int = 0
    num_classes: int = 0
    hidden_dim: int = 64
    heads: int = 1
    dropout: float = 0.5
    pooling: str = "mean"  # mean | max | sum | attention
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# AUDIO / SPEECH
# ═══════════════════════════════════════════════════════════════

@dataclass
class AudioDef(ASTNode):
    name: str = ""
    task: str = "classification"  # classification | asr | tts | separation | enhancement
    model_type: str = "wav2vec2"  # wav2vec2 | whisper | hubert | speecht5 | conformer
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 512
    data_ref: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# MULTIMODAL
# ═══════════════════════════════════════════════════════════════

@dataclass
class MultimodalDef(ASTNode):
    name: str = ""
    task: str = "vqa"  # vqa | captioning | retrieval | generation | grounding
    modalities: list = field(default_factory=list)  # ["image", "text", "audio"]
    vision_encoder: Optional[str] = None
    text_encoder: Optional[str] = None
    audio_encoder: Optional[str] = None
    fusion_method: str = "cross_attention"  # cross_attention | concat | late_fusion | early_fusion
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DISTILLATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistillDef(ASTNode):
    name: str = ""
    teacher: str = ""
    student: str = ""
    method: str = "kd"  # kd | attention | feature | relation | self
    temperature: float = 4.0
    alpha: float = 0.7
    data_ref: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# QUANTIZATION (standalone block)
# ═══════════════════════════════════════════════════════════════

@dataclass
class QuantizeDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    method: str = "dynamic"  # dynamic | static | qat | gptq | awq | bnb
    dtype: str = "int8"  # int8 | int4 | fp16 | bf16
    calibration_data: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# MONITORING / MLOPS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MonitorDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    backend: str = "wandb"  # wandb | mlflow | tensorboard | neptune | comet
    metrics: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    drift_detection: bool = False
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# SERVE (API serving block)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ServeDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    framework: str = "fastapi"  # fastapi | flask | gradio | streamlit | triton | torchserve
    endpoint: str = "/predict"
    host: str = "0.0.0.0"
    port: int = 8000
    batch: bool = False
    max_batch_size: int = 32
    timeout: int = 30
    auth: Optional[str] = None
    cors: bool = True
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TEST (model testing block)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    tests: list = field(default_factory=list)  # invariance, robustness, bias, etc.
    data_ref: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    metrics: list = field(default_factory=list)
    input_shape: Optional[list] = None
    num_warmup: int = 10
    num_runs: int = 100
    devices: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# AUGMENT (standalone augmentation pipeline)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AugmentDef(ASTNode):
    name: str = ""
    domain: str = "image"  # image | text | audio | tabular
    transforms: list = field(default_factory=list)
    probability: float = 1.0
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeatureDef(ASTNode):
    name: str = ""
    data_ref: Optional[str] = None
    operations: list = field(default_factory=list)
    target: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmbeddingDef(ASTNode):
    name: str = ""
    model: str = ""  # sentence-transformers, openai, cohere, etc.
    dim: int = 768
    pooling: str = "mean"
    normalize: bool = True
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════

@dataclass
class TokenizerDef(ASTNode):
    name: str = ""
    type: str = "bpe"  # bpe | wordpiece | unigram | sentencepiece | tiktoken
    vocab_size: int = 32000
    special_tokens: list = field(default_factory=list)
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# CALLBACK
# ═══════════════════════════════════════════════════════════════

@dataclass
class CallbackDef(ASTNode):
    name: str = ""
    trigger: str = "epoch_end"  # epoch_end | batch_end | train_start | train_end | eval_end
    actions: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# METRIC (custom metric definition)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MetricDef(ASTNode):
    name: str = ""
    formula: Optional[str] = None
    higher_is_better: bool = True
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# RAG (Retrieval-Augmented Generation)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RAGDef(ASTNode):
    name: str = ""
    retriever: str = "faiss"  # faiss | chromadb | pinecone | weaviate | milvus
    generator: str = ""
    embedding_model: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# AGENT (LLM Agent)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentDef(ASTNode):
    name: str = ""
    llm: str = ""
    tools: list = field(default_factory=list)
    memory: Optional[str] = None  # buffer | summary | vector
    max_iterations: int = 10
    system_prompt: Optional[str] = None
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# FEDERATED LEARNING
# ═══════════════════════════════════════════════════════════════

@dataclass
class FederatedDef(ASTNode):
    name: str = ""
    model_ref: str = ""
    num_clients: int = 10
    rounds: int = 100
    strategy: str = "fedavg"  # fedavg | fedprox | scaffold | fedadam
    client_optimizer: Any = None
    server_optimizer: Any = None
    fraction_fit: float = 0.5
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# AUTOML
# ═══════════════════════════════════════════════════════════════

@dataclass
class AutoMLDef(ASTNode):
    name: str = ""
    task: str = "classification"  # classification | regression | time_series | nlp
    data_ref: Optional[str] = None
    target_column: str = ""
    time_budget: int = 3600
    metric: str = "accuracy"
    framework: str = "auto"  # auto | autogluon | h2o | tpot | flaml
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# PYTHON ESCAPE
# ═══════════════════════════════════════════════════════════════

@dataclass
class PythonBlock(ASTNode):
    """@python: ... raw Python code block."""
    code: str = ""


# ═══════════════════════════════════════════════════════════════
# TOP-LEVEL PROGRAM
# ═══════════════════════════════════════════════════════════════

@dataclass
class Program(ASTNode):
    """Root node containing all top-level definitions."""
    imports: list = field(default_factory=list)
    definitions: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# MODULE SYSTEM (IMPORTS)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AxonImport(ASTNode):
    """Represents an import statement in Axon.
    
    Supports:
      import Model from "./other.axon"         -> import_style='named', names=['Model']
      import { Model, DataLoader } from "..."  -> import_style='named', names=[...]
      import * from "./all.axon"               -> import_style='wildcard'
      import numpy as np                       -> import_style='python', module='numpy', alias='np'
    """
    import_style: str = "named"   # 'named' | 'wildcard' | 'python'
    names: list = field(default_factory=list)   # list of str names to import
    source_path: Optional[str] = None           # './path/to/file.axon'
    module: str = ""                            # for python-style imports
    alias: Optional[str] = None                 # 'as' alias
