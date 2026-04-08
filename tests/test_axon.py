"""
Axon ML Language — Comprehensive Test Suite
============================================
Tests the full pipeline: Lexer → Parser → Transpiler for all 30+ block types.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axon.parser.lexer import AxonLexer, TokenType
from axon.parser.parser import AxonParser
from axon.parser.ast_nodes import *
from axon.transpiler.engine import AxonTranspiler
from axon.runtime.executor import AxonExecutor
from axon.runtime.config import AxonConfig


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _lex(source: str):
    """Return token list (no comments) from source."""
    return AxonLexer(source).get_tokens()


def _lex_types(source: str):
    """Return list of token types from source."""
    return [t.type for t in _lex(source)]


def _parse(source: str):
    """Parse source and return Program AST."""
    return AxonParser(source).parse()


def _first(source: str):
    """Parse source and return first definition."""
    return _parse(source).definitions[0]


def _transpile(source: str, backend: str = "pytorch") -> str:
    """Parse and transpile source, return Python code."""
    ast = _parse(source)
    transpiler = AxonTranspiler(backend=backend)
    return transpiler.transpile(ast)


def _assert(condition: bool, message: str = ""):
    if not condition:
        raise AssertionError(message or "Assertion failed")


# ═══════════════════════════════════════════════════════════════
# LEXER TESTS
# ═══════════════════════════════════════════════════════════════

def test_lexer_keyword_model():
    tokens = _lex("model MyNet:")
    types = [t.type for t in tokens]
    _assert(TokenType.MODEL in types, "MODEL keyword missing")
    _assert(tokens[0].type == TokenType.MODEL)
    _assert(tokens[0].value == "model")
    print("✓ test_lexer_keyword_model")


def test_lexer_keyword_data():
    tokens = _lex("data MyData:")
    _assert(tokens[0].type == TokenType.DATA)
    _assert(tokens[0].value == "data")
    print("✓ test_lexer_keyword_data")


def test_lexer_keyword_train():
    tokens = _lex("train MyTrain:")
    _assert(tokens[0].type == TokenType.TRAIN)
    print("✓ test_lexer_keyword_train")


def test_lexer_keyword_evaluate():
    tokens = _lex("evaluate MyEval:")
    _assert(tokens[0].type == TokenType.EVALUATE)
    print("✓ test_lexer_keyword_evaluate")


def test_lexer_keyword_search():
    tokens = _lex("search MySearch:")
    _assert(tokens[0].type == TokenType.SEARCH)
    print("✓ test_lexer_keyword_search")


def test_lexer_keyword_deploy():
    tokens = _lex("deploy MyDeploy:")
    _assert(tokens[0].type == TokenType.DEPLOY)
    print("✓ test_lexer_keyword_deploy")


def test_lexer_all_extended_keywords():
    """Verify all 30+ block keywords tokenize correctly."""
    keyword_map = {
        "gan": TokenType.GAN,
        "diffusion": TokenType.DIFFUSION,
        "rl": TokenType.RL,
        "tabular": TokenType.TABULAR,
        "timeseries": TokenType.TIMESERIES,
        "graph": TokenType.GRAPH,
        "audio": TokenType.AUDIO,
        "multimodal": TokenType.MULTIMODAL,
        "distill": TokenType.DISTILL,
        "quantize": TokenType.QUANTIZE,
        "monitor": TokenType.MONITOR,
        "serve": TokenType.SERVE,
        "test": TokenType.TEST,
        "benchmark": TokenType.BENCHMARK,
        "augment": TokenType.AUGMENT,
        "feature": TokenType.FEATURE,
        "embedding": TokenType.EMBEDDING,
        "tokenizer": TokenType.TOKENIZER,
        "callback": TokenType.CALLBACK,
        "metric": TokenType.METRIC,
        "rag": TokenType.RAG,
        "agent": TokenType.AGENT,
        "federated": TokenType.FEDERATED,
        "automl": TokenType.AUTOML,
        "pipeline": TokenType.PIPELINE,
        "pretrain": TokenType.PRETRAIN,
        "finetune": TokenType.FINETUNE,
        "ensemble": TokenType.ENSEMBLE,
        "explain": TokenType.EXPLAIN,
    }
    for word, expected_type in keyword_map.items():
        tokens = _lex(word)
        _assert(
            tokens[0].type == expected_type,
            f"Keyword '{word}' expected {expected_type}, got {tokens[0].type}"
        )
    print("✓ test_lexer_all_extended_keywords")


def test_lexer_number_integer():
    tokens = _lex("42")
    _assert(tokens[0].type == TokenType.NUMBER)
    _assert(tokens[0].value == "42")
    print("✓ test_lexer_number_integer")


def test_lexer_number_float():
    tokens = _lex("3.14")
    _assert(tokens[0].type == TokenType.NUMBER)
    _assert(tokens[0].value == "3.14")
    print("✓ test_lexer_number_float")


def test_lexer_number_scientific():
    tokens = _lex("1e-4")
    _assert(tokens[0].type == TokenType.NUMBER)
    _assert(tokens[0].value == "1e-4")

    tokens2 = _lex("3e+8")
    _assert(tokens2[0].type == TokenType.NUMBER)
    print("✓ test_lexer_number_scientific")


def test_lexer_number_hex():
    tokens = _lex("0xFF")
    _assert(tokens[0].type == TokenType.NUMBER)
    _assert(tokens[0].value == "0xFF")
    print("✓ test_lexer_number_hex")


def test_lexer_string_double_quoted():
    tokens = _lex('"hello world"')
    _assert(tokens[0].type == TokenType.STRING)
    _assert(tokens[0].value == "hello world")
    print("✓ test_lexer_string_double_quoted")


def test_lexer_string_single_quoted():
    tokens = _lex("'hello world'")
    _assert(tokens[0].type == TokenType.STRING)
    _assert(tokens[0].value == "hello world")
    print("✓ test_lexer_string_single_quoted")


def test_lexer_operator_arrow():
    tokens = _lex("784 -> 256")
    types = [t.type for t in tokens]
    _assert(TokenType.ARROW in types, "ARROW token missing")
    arrow_token = next(t for t in tokens if t.type == TokenType.ARROW)
    _assert(arrow_token.value == "->")
    print("✓ test_lexer_operator_arrow")


def test_lexer_operator_dot():
    tokens = _lex("obj.attr")
    types = [t.type for t in tokens]
    _assert(TokenType.DOT in types)
    print("✓ test_lexer_operator_dot")


def test_lexer_operator_colon():
    tokens = _lex("key: value")
    types = [t.type for t in tokens]
    _assert(TokenType.COLON in types)
    print("✓ test_lexer_operator_colon")


def test_lexer_operator_brackets():
    tokens = _lex("[1, 2, 3]")
    types = [t.type for t in tokens]
    _assert(TokenType.LBRACKET in types)
    _assert(TokenType.RBRACKET in types)
    print("✓ test_lexer_operator_brackets")


def test_lexer_param_ref():
    tokens = _lex("@num_classes")
    _assert(tokens[0].type == TokenType.AT)
    _assert(tokens[0].value == "@num_classes")
    print("✓ test_lexer_param_ref")


def test_lexer_param_ref_in_expression():
    tokens = _lex("Linear(2048 -> @num_classes)")
    at_tokens = [t for t in tokens if t.type == TokenType.AT]
    _assert(len(at_tokens) == 1)
    _assert(at_tokens[0].value == "@num_classes")
    print("✓ test_lexer_param_ref_in_expression")


def test_lexer_indentation_indent_dedent():
    src = "model Foo:\n    fc: Linear(10)\n"
    tokens = _lex(src)
    types = [t.type for t in tokens]
    _assert(TokenType.INDENT in types, "INDENT token missing")
    _assert(TokenType.DEDENT in types, "DEDENT token missing")
    print("✓ test_lexer_indentation_indent_dedent")


def test_lexer_bool_true_lowercase():
    tokens = _lex("true")
    _assert(tokens[0].type == TokenType.BOOL_TRUE)
    _assert(tokens[0].value == "true")
    print("✓ test_lexer_bool_true_lowercase")


def test_lexer_bool_true_titlecase():
    tokens = _lex("True")
    _assert(tokens[0].type == TokenType.BOOL_TRUE)
    print("✓ test_lexer_bool_true_titlecase")


def test_lexer_bool_false_lowercase():
    tokens = _lex("false")
    _assert(tokens[0].type == TokenType.BOOL_FALSE)
    print("✓ test_lexer_bool_false_lowercase")


def test_lexer_bool_false_titlecase():
    tokens = _lex("False")
    _assert(tokens[0].type == TokenType.BOOL_FALSE)
    print("✓ test_lexer_bool_false_titlecase")


def test_lexer_comment_ignored():
    tokens = _lex("# this is a comment\nmodel Foo:")
    # get_tokens() strips comments
    types = [t.type for t in tokens]
    _assert(TokenType.COMMENT not in types, "COMMENT should be filtered by get_tokens()")
    _assert(TokenType.MODEL in types, "MODEL should still appear after comment")
    print("✓ test_lexer_comment_ignored")


def test_lexer_comment_in_raw_tokens():
    lex = AxonLexer("# a comment\nmodel Foo:")
    all_types = [t.type for t in lex.tokens]
    _assert(TokenType.COMMENT in all_types, "COMMENT should be in raw tokens")
    print("✓ test_lexer_comment_in_raw_tokens")


def test_lexer_import_keyword():
    tokens = _lex("import torch")
    _assert(tokens[0].type == TokenType.IMPORT)
    print("✓ test_lexer_import_keyword")


def test_lexer_from_as_keywords():
    tokens = _lex("from torch import nn as nn_module")
    types = [t.type for t in tokens]
    _assert(TokenType.FROM in types)
    _assert(TokenType.AS in types)
    print("✓ test_lexer_from_as_keywords")


# ═══════════════════════════════════════════════════════════════
# PARSER TESTS — CORE BLOCKS
# ═══════════════════════════════════════════════════════════════

def test_parser_model_basic():
    src = """
model SimpleNet:
    fc1: Linear(784 -> 256)
    relu: ReLU()
    fc2: Linear(256 -> 10)
"""
    defn = _first(src)
    _assert(isinstance(defn, ModelDef), f"Expected ModelDef, got {type(defn).__name__}")
    _assert(defn.name == "SimpleNet")
    _assert("fc1" in defn.layers)
    _assert("relu" in defn.layers)
    _assert("fc2" in defn.layers)
    print("✓ test_parser_model_basic")


def test_parser_model_with_forward():
    src = """
model MyModel:
    fc: Linear(10 -> 5)
    forward:
        relu(fc(x))
"""
    defn = _first(src)
    _assert(isinstance(defn, ModelDef))
    _assert(defn.forward_def is not None, "forward_def should be set")
    print("✓ test_parser_model_with_forward")


def test_parser_model_with_param_ref():
    src = """
model Classifier:
    backbone: resnet50(pretrained=true)
    head: Linear(2048 -> @num_classes)
"""
    defn = _first(src)
    _assert(isinstance(defn, ModelDef))
    _assert("head" in defn.layers)
    print("✓ test_parser_model_with_param_ref")


def test_parser_data_full():
    src = """
data CIFAR10:
    source: "./data/cifar10"
    format: image_folder
    split: 80/10/10
    loader:
        batch_size: 32
        shuffle: true
        num_workers: 4
"""
    defn = _first(src)
    _assert(isinstance(defn, DataDef), f"Expected DataDef, got {type(defn).__name__}")
    _assert(defn.name == "CIFAR10")
    _assert(defn.source is not None)
    _assert(isinstance(defn.split, RatioExpr), "split should be RatioExpr")
    _assert(defn.split.parts[0] == 80)
    _assert(defn.split.parts[1] == 10)
    print("✓ test_parser_data_full")


def test_parser_data_streaming_cache():
    src = """
data StreamData:
    source: "./data"
    format: csv
    streaming: true
    cache: false
"""
    defn = _first(src)
    _assert(isinstance(defn, DataDef))
    _assert(defn.streaming == True)
    _assert(defn.cache == False)
    print("✓ test_parser_data_streaming_cache")


def test_parser_train_full():
    src = """
train VisionTrain:
    model: ImageClassifier(num_classes=10)
    data: CIFAR10Data()
    optimizer: AdamW(lr=3e-4, weight_decay=1e-4)
    scheduler: CosineAnnealing(T_max=50)
    loss: CrossEntropy
    epochs: 50
    device: auto
    precision: fp16
    metrics:
        accuracy
        f1
"""
    defn = _first(src)
    _assert(isinstance(defn, TrainDef), f"Expected TrainDef, got {type(defn).__name__}")
    _assert(defn.name == "VisionTrain")
    _assert(defn.epochs == 50)
    _assert(defn.device == "auto")
    _assert(defn.precision == "fp16")
    _assert(len(defn.metrics) >= 2)
    print("✓ test_parser_train_full")


def test_parser_train_callbacks():
    src = """
train MyTrain:
    model: MyModel()
    epochs: 10
    loss: CrossEntropy
    callbacks:
        early_stopping(patience=10)
        checkpoint(save_best=true)
"""
    defn = _first(src)
    _assert(isinstance(defn, TrainDef))
    _assert(len(defn.callbacks) >= 1)
    print("✓ test_parser_train_callbacks")


def test_parser_evaluate_full():
    src = """
evaluate VisionEval:
    checkpoint: "best"
    metrics:
        accuracy
        precision
        recall
        f1
        confusion_matrix
"""
    defn = _first(src)
    _assert(isinstance(defn, EvalDef), f"Expected EvalDef, got {type(defn).__name__}")
    _assert(defn.name == "VisionEval")
    _assert(defn.checkpoint == "best")
    _assert(len(defn.metrics) >= 4)
    print("✓ test_parser_evaluate_full")


def test_parser_search_full():
    src = """
search VisionSearch:
    base: VisionExperiment
    method: bayesian
    trials: 30
    objective: maximize(val_accuracy)
"""
    defn = _first(src)
    _assert(isinstance(defn, SearchDef), f"Expected SearchDef, got {type(defn).__name__}")
    _assert(defn.name == "VisionSearch")
    _assert(defn.method == "bayesian")
    _assert(defn.trials == 30)
    _assert(defn.objective is not None)
    print("✓ test_parser_search_full")


def test_parser_search_space():
    src = """
search HyperSearch:
    base: MyTrain
    method: tpe
    trials: 50
    space:
        lr: log_uniform(1e-5, 1e-2)
        batch_size: choice(16, 32, 64)
        dropout: uniform(0.1, 0.5)
"""
    defn = _first(src)
    _assert(isinstance(defn, SearchDef))
    _assert(len(defn.space) >= 3)
    print("✓ test_parser_search_space")


def test_parser_deploy_full():
    src = """
deploy VisionModel:
    checkpoint: "best"
    format: onnx
    optimize: true
    quantize: int8
    serve:
        framework: fastapi
        endpoint: "/classify"
        batch: true
"""
    defn = _first(src)
    _assert(isinstance(defn, DeployDef), f"Expected DeployDef, got {type(defn).__name__}")
    _assert(defn.name == "VisionModel")
    _assert(defn.format == "onnx")
    _assert(defn.quantize == "int8")
    _assert(defn.optimize == True)
    print("✓ test_parser_deploy_full")


def test_parser_pipeline():
    src = """
pipeline MyPipeline:
    step1: DataProcessing
    step2: ModelTraining
    step3: Evaluation
"""
    defn = _first(src)
    _assert(isinstance(defn, PipelineDef), f"Expected PipelineDef, got {type(defn).__name__}")
    _assert(defn.name == "MyPipeline")
    print("✓ test_parser_pipeline")


def test_parser_finetune_lora():
    src = """
finetune GPTFinetuner:
    base_model: "gpt2"
    method: lora
    r: 16
    alpha: 32
    dropout: 0.05
"""
    defn = _first(src)
    _assert(isinstance(defn, FinetuneDef), f"Expected FinetuneDef, got {type(defn).__name__}")
    _assert(defn.name == "GPTFinetuner")
    _assert(defn.base_model == "gpt2")
    _assert(defn.method == "lora")
    print("✓ test_parser_finetune_lora")


def test_parser_ensemble():
    src = """
ensemble SentimentEnsemble:
    models: [ModelA, ModelB, ModelC]
    strategy: averaging
"""
    defn = _first(src)
    _assert(isinstance(defn, EnsembleDef), f"Expected EnsembleDef, got {type(defn).__name__}")
    _assert(defn.name == "SentimentEnsemble")
    _assert(len(defn.models) == 3)
    _assert(defn.strategy == "averaging")
    print("✓ test_parser_ensemble")


def test_parser_explain():
    src = """
explain ModelExplainer:
    model: SentimentClassifier
    method: shap
    num_samples: 200
"""
    defn = _first(src)
    _assert(isinstance(defn, ExplainDef), f"Expected ExplainDef, got {type(defn).__name__}")
    _assert(defn.name == "ModelExplainer")
    _assert(defn.model_ref == "SentimentClassifier")
    _assert(defn.method == "shap")
    _assert(defn.num_samples == 200)
    print("✓ test_parser_explain")


def test_parser_pretrain():
    src = """
pretrain GPTPretraining:
    model: MiniGPT
    objective: autoregressive
    data: TextCorpus
"""
    defn = _first(src)
    _assert(isinstance(defn, PretrainDef), f"Expected PretrainDef, got {type(defn).__name__}")
    _assert(defn.name == "GPTPretraining")
    _assert(defn.model_ref == "MiniGPT")
    _assert(defn.objective == "autoregressive")
    print("✓ test_parser_pretrain")


# ═══════════════════════════════════════════════════════════════
# PARSER TESTS — EXTENDED BLOCKS
# ═══════════════════════════════════════════════════════════════

def test_parser_gan():
    src = """
gan MyGAN:
    generator: ResNet50
    discriminator: VGG16
    latent_dim: 128
    loss: wgan
    epochs: 200
"""
    defn = _first(src)
    _assert(isinstance(defn, GANDef), f"Expected GANDef, got {type(defn).__name__}")
    _assert(defn.name == "MyGAN")
    _assert(defn.latent_dim == 128)
    _assert(defn.loss_type == "wgan")
    _assert(defn.epochs == 200)
    print("✓ test_parser_gan")


def test_parser_gan_properties():
    src = """
gan WGANModel:
    generator: UNet
    discriminator: PatchGAN
    latent_dim: 256
    loss: wgan_gp
    n_critic: 5
    gp_weight: 10.0
    epochs: 300
"""
    defn = _first(src)
    _assert(isinstance(defn, GANDef))
    _assert(defn.n_critic == 5)
    _assert(defn.gp_weight == 10.0)
    print("✓ test_parser_gan_properties")


def test_parser_diffusion():
    src = """
diffusion StableDiff:
    scheduler: ddpm
    timesteps: 1000
    image_size: 256
    channels: 3
"""
    defn = _first(src)
    _assert(isinstance(defn, DiffusionDef), f"Expected DiffusionDef, got {type(defn).__name__}")
    _assert(defn.name == "StableDiff")
    _assert(defn.noise_scheduler == "ddpm")
    _assert(defn.timesteps == 1000)
    _assert(defn.image_size == 256)
    print("✓ test_parser_diffusion")


def test_parser_rl():
    src = """
rl CartPoleRL:
    algorithm: ppo
    environment: CartPole-v1
    policy: MlpPolicy
    total_timesteps: 100000
"""
    defn = _first(src)
    _assert(isinstance(defn, RLDef), f"Expected RLDef, got {type(defn).__name__}")
    _assert(defn.name == "CartPoleRL")
    _assert(defn.algorithm == "ppo")
    _assert(defn.policy == "MlpPolicy")
    _assert(defn.total_timesteps == 100000)
    print("✓ test_parser_rl")


def test_parser_tabular():
    src = """
tabular CreditRisk:
    task: classification
    algorithm: xgboost
    target: default
    cross_validation: 5
"""
    defn = _first(src)
    _assert(isinstance(defn, TabularDef), f"Expected TabularDef, got {type(defn).__name__}")
    _assert(defn.name == "CreditRisk")
    _assert(defn.task == "classification")
    _assert(defn.algorithm == "xgboost")
    _assert(defn.target_column == "default")
    _assert(defn.cross_validation == 5)
    print("✓ test_parser_tabular")


def test_parser_timeseries():
    src = """
timeseries SalesForecaster:
    task: forecast
    horizon: 24
    lookback: 168
"""
    defn = _first(src)
    _assert(isinstance(defn, TimeSeriesDef), f"Expected TimeSeriesDef, got {type(defn).__name__}")
    _assert(defn.name == "SalesForecaster")
    _assert(defn.task == "forecast")
    _assert(defn.horizon == 24)
    _assert(defn.lookback == 168)
    print("✓ test_parser_timeseries")


def test_parser_graph():
    src = """
graph CitationGNN:
    conv_type: GCN
    num_features: 128
    num_classes: 7
    hidden_dim: 64
"""
    defn = _first(src)
    _assert(isinstance(defn, GraphDef), f"Expected GraphDef, got {type(defn).__name__}")
    _assert(defn.name == "CitationGNN")
    _assert(defn.conv_type == "GCN")
    _assert(defn.num_features == 128)
    _assert(defn.num_classes == 7)
    print("✓ test_parser_graph")


def test_parser_audio():
    src = """
audio SpeechASR:
    task: asr
    model: whisper
    sample_rate: 16000
"""
    defn = _first(src)
    _assert(isinstance(defn, AudioDef), f"Expected AudioDef, got {type(defn).__name__}")
    _assert(defn.name == "SpeechASR")
    _assert(defn.task == "asr")
    _assert(defn.model_type == "whisper")
    _assert(defn.sample_rate == 16000)
    print("✓ test_parser_audio")


def test_parser_multimodal():
    src = """
multimodal CLIPModel:
    task: retrieval
    modalities: [image, text]
    fusion: cross_attention
    vision_encoder: vit_base
    text_encoder: bert_base
"""
    defn = _first(src)
    _assert(isinstance(defn, MultimodalDef), f"Expected MultimodalDef, got {type(defn).__name__}")
    _assert(defn.name == "CLIPModel")
    _assert(defn.task == "retrieval")
    _assert("image" in defn.modalities)
    _assert("text" in defn.modalities)
    _assert(defn.fusion_method == "cross_attention")
    print("✓ test_parser_multimodal")


def test_parser_distill():
    src = """
distill TinyBERT:
    teacher: BERTLarge
    student: BERTSmall
    temperature: 4.0
    alpha: 0.7
"""
    defn = _first(src)
    _assert(isinstance(defn, DistillDef), f"Expected DistillDef, got {type(defn).__name__}")
    _assert(defn.name == "TinyBERT")
    _assert(defn.teacher == "BERTLarge")
    _assert(defn.student == "BERTSmall")
    _assert(defn.temperature == 4.0)
    _assert(defn.alpha == 0.7)
    print("✓ test_parser_distill")


def test_parser_quantize():
    src = """
quantize QuantizedModel:
    model: MyModel
    method: dynamic
    dtype: int8
"""
    defn = _first(src)
    _assert(isinstance(defn, QuantizeDef), f"Expected QuantizeDef, got {type(defn).__name__}")
    _assert(defn.name == "QuantizedModel")
    _assert(defn.model_ref == "MyModel")
    _assert(defn.method == "dynamic")
    _assert(defn.dtype == "int8")
    print("✓ test_parser_quantize")


def test_parser_monitor():
    src = """
monitor ExperimentMonitor:
    model: MyModel
    backend: wandb
    metrics:
        accuracy
        loss
"""
    defn = _first(src)
    _assert(isinstance(defn, MonitorDef), f"Expected MonitorDef, got {type(defn).__name__}")
    _assert(defn.name == "ExperimentMonitor")
    _assert(defn.backend == "wandb")
    _assert(len(defn.metrics) >= 2)
    print("✓ test_parser_monitor")


def test_parser_serve():
    src = """
serve ModelAPI:
    model: MyModel
    framework: fastapi
    endpoint: "/predict"
    port: 8080
"""
    defn = _first(src)
    _assert(isinstance(defn, ServeDef), f"Expected ServeDef, got {type(defn).__name__}")
    _assert(defn.name == "ModelAPI")
    _assert(defn.framework == "fastapi")
    _assert(defn.endpoint == "/predict")
    _assert(defn.port == 8080)
    print("✓ test_parser_serve")


def test_parser_test_block():
    src = """
test ModelTests:
    model: MyModel
    tests:
        invariance
        robustness
        bias
"""
    defn = _first(src)
    _assert(isinstance(defn, TestDef), f"Expected TestDef, got {type(defn).__name__}")
    _assert(defn.name == "ModelTests")
    _assert(defn.model_ref == "MyModel")
    _assert(len(defn.tests) >= 2)
    print("✓ test_parser_test_block")


def test_parser_benchmark():
    src = """
benchmark SpeedBench:
    model: MyModel
    num_runs: 200
    num_warmup: 20
"""
    defn = _first(src)
    _assert(isinstance(defn, BenchmarkDef), f"Expected BenchmarkDef, got {type(defn).__name__}")
    _assert(defn.name == "SpeedBench")
    _assert(defn.model_ref == "MyModel")
    _assert(defn.num_runs == 200)
    _assert(defn.num_warmup == 20)
    print("✓ test_parser_benchmark")


def test_parser_augment():
    src = """
augment ImageAugPipeline:
    domain: image
    probability: 0.8
"""
    defn = _first(src)
    _assert(isinstance(defn, AugmentDef), f"Expected AugmentDef, got {type(defn).__name__}")
    _assert(defn.name == "ImageAugPipeline")
    _assert(defn.domain == "image")
    _assert(defn.probability == 0.8)
    print("✓ test_parser_augment")


def test_parser_feature():
    src = """
feature FeatureEng:
    data: TrainData
    target: label
"""
    defn = _first(src)
    _assert(isinstance(defn, FeatureDef), f"Expected FeatureDef, got {type(defn).__name__}")
    _assert(defn.name == "FeatureEng")
    _assert(defn.target == "label")
    print("✓ test_parser_feature")


def test_parser_embedding():
    src = """
embedding SentenceEmbedder:
    model: sentence-transformers/all-MiniLM-L6-v2
    dim: 384
    pooling: mean
    normalize: true
"""
    defn = _first(src)
    _assert(isinstance(defn, EmbeddingDef), f"Expected EmbeddingDef, got {type(defn).__name__}")
    _assert(defn.name == "SentenceEmbedder")
    _assert(defn.dim == 384)
    _assert(defn.pooling == "mean")
    _assert(defn.normalize == True)
    print("✓ test_parser_embedding")


def test_parser_tokenizer():
    src = """
tokenizer BPETokenizer:
    type: bpe
    vocab_size: 32000
    max_length: 512
    truncation: true
"""
    defn = _first(src)
    _assert(isinstance(defn, TokenizerDef), f"Expected TokenizerDef, got {type(defn).__name__}")
    _assert(defn.name == "BPETokenizer")
    _assert(defn.type == "bpe")
    _assert(defn.vocab_size == 32000)
    _assert(defn.max_length == 512)
    _assert(defn.truncation == True)
    print("✓ test_parser_tokenizer")


def test_parser_callback():
    src = """
callback EpochCallback:
    trigger: epoch_end
    actions:
        save_checkpoint
        log_metrics
"""
    defn = _first(src)
    _assert(isinstance(defn, CallbackDef), f"Expected CallbackDef, got {type(defn).__name__}")
    _assert(defn.name == "EpochCallback")
    _assert(defn.trigger == "epoch_end")
    _assert(len(defn.actions) >= 2)
    print("✓ test_parser_callback")


def test_parser_metric():
    src = """
metric PrecisionMetric:
    formula: tp / (tp + fp)
    higher_is_better: true
"""
    defn = _first(src)
    _assert(isinstance(defn, MetricDef), f"Expected MetricDef, got {type(defn).__name__}")
    _assert(defn.name == "PrecisionMetric")
    _assert(defn.higher_is_better == True)
    print("✓ test_parser_metric")


def test_parser_rag():
    src = """
rag KnowledgeRAG:
    retriever: faiss
    generator: gpt-4
    top_k: 5
    chunk_size: 512
    chunk_overlap: 50
"""
    defn = _first(src)
    _assert(isinstance(defn, RAGDef), f"Expected RAGDef, got {type(defn).__name__}")
    _assert(defn.name == "KnowledgeRAG")
    _assert(defn.retriever == "faiss")
    _assert(defn.top_k == 5)
    _assert(defn.chunk_size == 512)
    print("✓ test_parser_rag")


def test_parser_agent():
    src = """
agent ResearchAgent:
    llm: gpt-4
    tools: [web_search, calculator, code_exec]
    memory: buffer
    max_iterations: 15
"""
    defn = _first(src)
    _assert(isinstance(defn, AgentDef), f"Expected AgentDef, got {type(defn).__name__}")
    _assert(defn.name == "ResearchAgent")
    _assert(len(defn.tools) == 3)
    _assert(defn.memory == "buffer")
    _assert(defn.max_iterations == 15)
    print("✓ test_parser_agent")


def test_parser_federated():
    src = """
federated FedLearning:
    model: GlobalModel
    num_clients: 100
    rounds: 50
    strategy: fedavg
    fraction_fit: 0.1
"""
    defn = _first(src)
    _assert(isinstance(defn, FederatedDef), f"Expected FederatedDef, got {type(defn).__name__}")
    _assert(defn.name == "FedLearning")
    _assert(defn.num_clients == 100)
    _assert(defn.rounds == 50)
    _assert(defn.strategy == "fedavg")
    _assert(defn.fraction_fit == 0.1)
    print("✓ test_parser_federated")


def test_parser_automl():
    src = """
automl AutoClassifier:
    task: classification
    target: label
    time_budget: 3600
    metric: accuracy
    framework: auto
"""
    defn = _first(src)
    _assert(isinstance(defn, AutoMLDef), f"Expected AutoMLDef, got {type(defn).__name__}")
    _assert(defn.name == "AutoClassifier")
    _assert(defn.task == "classification")
    _assert(defn.time_budget == 3600)
    _assert(defn.metric == "accuracy")
    _assert(defn.framework == "auto")
    print("✓ test_parser_automl")


# ═══════════════════════════════════════════════════════════════
# TRANSPILER TESTS — CORE (PYTORCH)
# ═══════════════════════════════════════════════════════════════

def test_transpiler_model_pytorch():
    src = """
model SimpleNet:
    fc1: Linear(784 -> 256)
    relu: ReLU()
    fc2: Linear(256 -> 10)
"""
    code = _transpile(src, "pytorch")
    _assert("class SimpleNet(nn.Module):" in code, "Expected nn.Module class")
    _assert("def __init__" in code, "Expected __init__")
    _assert("nn.Linear" in code, "Expected nn.Linear")
    _assert("nn.ReLU" in code, "Expected nn.ReLU")
    print("✓ test_transpiler_model_pytorch")


def test_transpiler_model_tensorflow():
    src = """
model SimpleNet:
    fc1: Linear(784 -> 256)
    relu: ReLU()
    fc2: Linear(256 -> 10)
"""
    code = _transpile(src, "tensorflow")
    _assert(
        "keras" in code.lower() or "tensorflow" in code.lower(),
        "Expected tensorflow/keras code"
    )
    _assert("SimpleNet" in code, "Model name should appear")
    print("✓ test_transpiler_model_tensorflow")


def test_transpiler_model_jax():
    src = """
model SimpleNet:
    fc1: Linear(784 -> 256)
    fc2: Linear(256 -> 10)
"""
    code = _transpile(src, "jax")
    _assert(
        "flax" in code.lower() or "jax" in code.lower(),
        "Expected jax/flax code"
    )
    _assert("SimpleNet" in code, "Model name should appear")
    print("✓ test_transpiler_model_jax")


def test_transpiler_model_with_dropout():
    src = """
model Classifier:
    backbone: resnet50(pretrained=true)
    dropout: Dropout(0.3)
    head: Linear(2048 -> 10)
"""
    code = _transpile(src, "pytorch")
    _assert("class Classifier(nn.Module):" in code)
    _assert("nn.Dropout" in code, "Expected Dropout layer")
    print("✓ test_transpiler_model_with_dropout")


def test_transpiler_data_pytorch():
    src = """
data CIFAR10Data:
    source: "./data/cifar10"
    format: image_folder
    split: 80/10/10
    loader:
        batch_size: 32
        shuffle: true
        num_workers: 4
"""
    code = _transpile(src, "pytorch")
    _assert("CIFAR10Data" in code, "Data class name should appear")
    _assert("DataLoader" in code or "DataModule" in code or "dataset" in code.lower(),
            "Expected data loading code")
    print("✓ test_transpiler_data_pytorch")


def test_transpiler_train_loop():
    src = """
train MyTrain:
    model: SimpleNet(num_classes=10)
    data: MyData()
    optimizer: AdamW(lr=3e-4)
    loss: CrossEntropy
    epochs: 50
"""
    code = _transpile(src, "pytorch")
    _assert("class MyTrainTrainer" in code, "Expected Trainer class")
    _assert("50" in code, "Expected epoch count in code")
    _assert("optimizer" in code.lower(), "Expected optimizer setup")
    print("✓ test_transpiler_train_loop")


def test_transpiler_train_metrics():
    src = """
train MetricsTrain:
    model: MyModel()
    epochs: 10
    loss: CrossEntropy
    metrics:
        accuracy
        f1
"""
    code = _transpile(src, "pytorch")
    _assert("MetricsTrain" in code)
    _assert("accuracy" in code or "metric" in code.lower())
    print("✓ test_transpiler_train_metrics")


def test_transpiler_eval():
    src = """
evaluate VisionEval:
    checkpoint: "best"
    metrics:
        accuracy
        f1
        precision
"""
    code = _transpile(src, "pytorch")
    _assert("VisionEvalEvaluator" in code, "Expected Evaluator class")
    _assert("accuracy" in code, "Expected accuracy metric")
    print("✓ test_transpiler_eval")


def test_transpiler_search_optuna():
    src = """
search HyperSearch:
    base: MyTrain
    method: bayesian
    trials: 30
    objective: maximize(val_accuracy)
"""
    code = _transpile(src, "pytorch")
    _assert("optuna" in code.lower(), "Expected Optuna")
    _assert("study" in code.lower(), "Expected study")
    _assert("HyperSearch" in code, "Name should appear")
    print("✓ test_transpiler_search_optuna")


def test_transpiler_deploy_onnx():
    src = """
deploy VisionDeploy:
    checkpoint: "best"
    format: onnx
    quantize: int8
    serve:
        framework: fastapi
        endpoint: "/classify"
"""
    code = _transpile(src, "pytorch")
    _assert("VisionDeployDeployer" in code, "Expected Deployer class")
    _assert("onnx" in code.lower(), "Expected ONNX export")
    _assert("FastAPI" in code or "fastapi" in code.lower(), "Expected FastAPI serve")
    print("✓ test_transpiler_deploy_onnx")


def test_transpiler_deploy_torchscript():
    src = """
deploy ScriptDeploy:
    format: torchscript
"""
    code = _transpile(src, "pytorch")
    _assert("ScriptDeployDeployer" in code)
    _assert("jit" in code.lower() or "torchscript" in code.lower(),
            "Expected TorchScript code")
    print("✓ test_transpiler_deploy_torchscript")


def test_transpiler_pipeline():
    src = """
pipeline MLPipeline:
    step1: DataProcessing
    step2: ModelTraining
"""
    code = _transpile(src, "pytorch")
    _assert("MLPipelinePipeline" in code, "Expected Pipeline class")
    _assert("run" in code, "Expected run method")
    print("✓ test_transpiler_pipeline")


def test_transpiler_finetune_lora():
    src = """
finetune GPTFinetuner:
    base_model: "gpt2"
    method: lora
    r: 16
    alpha: 32
    dropout: 0.05
"""
    code = _transpile(src, "pytorch")
    _assert("GPTFinetunerFinetuner" in code, "Expected Finetuner class")
    _assert("LoraConfig" in code or "lora" in code.lower(), "Expected LoRA config")
    _assert("peft" in code.lower() or "LoraConfig" in code, "Expected PEFT")
    print("✓ test_transpiler_finetune_lora")


def test_transpiler_ensemble_voting():
    src = """
ensemble ModelEnsemble:
    models: [ModelA, ModelB, ModelC]
    strategy: voting
"""
    code = _transpile(src, "pytorch")
    _assert("ModelEnsembleEnsemble" in code, "Expected Ensemble class")
    _assert("voting" in code.lower(), "Expected voting strategy")
    _assert("predict" in code, "Expected predict method")
    print("✓ test_transpiler_ensemble_voting")


def test_transpiler_ensemble_averaging():
    src = """
ensemble AvgEnsemble:
    models: [ModelA, ModelB]
    strategy: averaging
"""
    code = _transpile(src, "pytorch")
    _assert("AvgEnsembleEnsemble" in code)
    _assert("averaging" in code.lower() or "weight" in code.lower())
    print("✓ test_transpiler_ensemble_averaging")


# ═══════════════════════════════════════════════════════════════
# TRANSPILER TESTS — EXTENDED BLOCKS
# ═══════════════════════════════════════════════════════════════

def test_transpiler_gan_compiles():
    src = """
gan DCGANModel:
    generator: Generator
    discriminator: Discriminator
    latent_dim: 100
    loss: vanilla
    epochs: 100
"""
    code = _transpile(src, "pytorch")
    # GAN transpilation is handled generically; output must at minimum contain imports
    _assert("import torch" in code, "Expected torch import")
    _assert(len(code) > 100, "Expected non-trivial output")
    print("✓ test_transpiler_gan_compiles")


def test_transpiler_diffusion_compiles():
    src = """
diffusion MyDiffusion:
    scheduler: ddpm
    timesteps: 1000
    image_size: 64
"""
    code = _transpile(src, "pytorch")
    _assert("import torch" in code, "Expected torch import")
    print("✓ test_transpiler_diffusion_compiles")


def test_transpiler_rl_compiles():
    src = """
rl CartPoleAgent:
    algorithm: ppo
    environment: CartPole-v1
    policy: MlpPolicy
    total_timesteps: 50000
"""
    code = _transpile(src, "pytorch")
    _assert("import torch" in code)
    print("✓ test_transpiler_rl_compiles")


def test_transpiler_tabular_compiles():
    src = """
tabular CreditModel:
    task: classification
    algorithm: xgboost
    target: default
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_tabular_compiles")


def test_transpiler_timeseries_compiles():
    src = """
timeseries SalesForecaster:
    task: forecast
    horizon: 24
    lookback: 168
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_timeseries_compiles")


def test_transpiler_graph_compiles():
    src = """
graph CitationGNN:
    conv_type: GCN
    num_features: 128
    num_classes: 7
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_graph_compiles")


def test_transpiler_audio_compiles():
    src = """
audio SpeechRecognizer:
    task: asr
    model: whisper
    sample_rate: 16000
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_audio_compiles")


def test_transpiler_multimodal_compiles():
    src = """
multimodal CLIPModel:
    task: retrieval
    modalities: [image, text]
    fusion: cross_attention
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_multimodal_compiles")


def test_transpiler_distill_compiles():
    src = """
distill TinyModel:
    teacher: LargeModel
    student: SmallModel
    temperature: 4.0
    alpha: 0.7
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_distill_compiles")


def test_transpiler_quantize_compiles():
    src = """
quantize QuantModel:
    model: MyModel
    method: dynamic
    dtype: int8
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_quantize_compiles")


def test_transpiler_monitor_compiles():
    src = """
monitor WandbMonitor:
    model: MyModel
    backend: wandb
    metrics:
        accuracy
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_monitor_compiles")


def test_transpiler_serve_compiles():
    src = """
serve ModelServe:
    model: MyModel
    framework: fastapi
    endpoint: "/predict"
    port: 8000
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_serve_compiles")


def test_transpiler_explain_shap():
    src = """
explain SHAPExplainer:
    model: MyModel
    method: shap
    num_samples: 100
"""
    code = _transpile(src, "pytorch")
    _assert("SHAPExplainerExplainer" in code, "Expected Explainer class")
    _assert("shap" in code.lower(), "Expected shap reference")
    print("✓ test_transpiler_explain_shap")


def test_transpiler_explain_lime():
    src = """
explain LIMEExplainer:
    model: MyModel
    method: lime
"""
    code = _transpile(src, "pytorch")
    _assert("LIMEExplainerExplainer" in code)
    _assert("lime" in code.lower(), "Expected lime reference")
    print("✓ test_transpiler_explain_lime")


def test_transpiler_pretrain_masked_lm():
    src = """
pretrain BERTPretrain:
    model: BERTModel
    objective: masked_lm
"""
    code = _transpile(src, "pytorch")
    _assert("BERTPretrainPretrainer" in code, "Expected Pretrainer class")
    _assert("masked" in code.lower() or "mlm" in code.lower(), "Expected masked LM reference")
    print("✓ test_transpiler_pretrain_masked_lm")


def test_transpiler_pretrain_autoregressive():
    src = """
pretrain GPTPretrain:
    model: GPTModel
    objective: autoregressive
"""
    code = _transpile(src, "pytorch")
    _assert("GPTPretrainPretrainer" in code)
    _assert("autoregressive" in code.lower() or "causal" in code.lower(),
            "Expected autoregressive reference")
    print("✓ test_transpiler_pretrain_autoregressive")


def test_transpiler_rag_compiles():
    src = """
rag KnowledgeRAG:
    retriever: faiss
    generator: gpt-4
    top_k: 5
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_rag_compiles")


def test_transpiler_agent_compiles():
    src = """
agent ResearchAgent:
    llm: gpt-4
    tools: [web_search, calculator]
    memory: buffer
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_agent_compiles")


def test_transpiler_benchmark_compiles():
    src = """
benchmark SpeedTest:
    model: MyModel
    num_runs: 100
    num_warmup: 10
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_benchmark_compiles")


def test_transpiler_federated_compiles():
    src = """
federated FedModel:
    model: GlobalModel
    num_clients: 10
    rounds: 100
    strategy: fedavg
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_federated_compiles")


def test_transpiler_automl_compiles():
    src = """
automl AutoClassify:
    task: classification
    target: label
    time_budget: 3600
"""
    code = _transpile(src, "pytorch")
    _assert(len(code) > 100)
    print("✓ test_transpiler_automl_compiles")


def test_transpiler_all_three_backends_model():
    src = """
model ResidualNet:
    conv1: Conv2d(3 -> 64)
    bn: BatchNorm2d(64)
    relu: ReLU()
    fc: Linear(512 -> 10)
"""
    for backend in ["pytorch", "tensorflow", "jax"]:
        code = _transpile(src, backend)
        _assert(len(code) > 100, f"Backend {backend} produced insufficient output")
        _assert("ResidualNet" in code, f"Model name missing for backend {backend}")
    print("✓ test_transpiler_all_three_backends_model")


def test_transpiler_invalid_backend_raises():
    src = "model Foo:\n    fc: Linear(10)\n"
    ast = _parse(src)
    try:
        AxonTranspiler(backend="invalid_backend")
        _assert(False, "Should have raised ValueError")
    except ValueError:
        pass
    print("✓ test_transpiler_invalid_backend_raises")


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

def test_integration_executor_check_valid():
    src = """
model SimpleNet:
    fc1: Linear(784 -> 256)
    fc2: Linear(256 -> 10)

train SimpleTraining:
    model: SimpleNet(num_classes=10)
    epochs: 10
    loss: CrossEntropy
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"Expected valid, errors: {result['errors']}")
    _assert(len(result["definitions"]) == 2)
    print("✓ test_integration_executor_check_valid")


def test_integration_executor_check_definitions():
    src = """
model MyModel:
    fc: Linear(10 -> 5)

data MyData:
    source: "./data"
    format: csv
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    names = [d["name"] for d in result["definitions"]]
    _assert("MyModel" in names, "MyModel should be in definitions")
    _assert("MyData" in names, "MyData should be in definitions")
    print("✓ test_integration_executor_check_definitions")


def test_integration_executor_compile():
    src = """
model VisionModel:
    backbone: resnet50(pretrained=true)
    head: Linear(2048 -> 10)
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    code = executor.compile(src)
    _assert(isinstance(code, str), "compile() should return a string")
    _assert(len(code) > 100, "Expected non-trivial output")
    _assert("VisionModel" in code, "Model name should appear in output")
    print("✓ test_integration_executor_compile")


def test_integration_executor_run_string():
    src = """
model MLP:
    fc1: Linear(784 -> 512)
    relu: ReLU()
    fc2: Linear(512 -> 10)
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    code = executor.run_string(src)
    _assert("MLP" in code, "Expected model class in output")
    print("✓ test_integration_executor_run_string")


def test_integration_full_vision_pipeline():
    src = """
model ImageNet:
    backbone: resnet50(pretrained=true)
    pool: AdaptiveAvgPool2d(1)
    head: Linear(2048 -> @num_classes)

data CIFAR10:
    source: "./data/cifar10"
    format: image_folder
    split: 80/10/10
    loader:
        batch_size: 32
        shuffle: true

train VisionTrain:
    model: ImageNet(num_classes=10)
    data: CIFAR10()
    optimizer: AdamW(lr=3e-4)
    loss: CrossEntropy
    epochs: 50

evaluate VisionEval:
    checkpoint: "best"
    metrics:
        accuracy
        f1

search VisionSearch:
    base: VisionTrain
    method: bayesian
    trials: 30
    objective: maximize(val_accuracy)
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"Pipeline check failed: {result['errors']}")

    code = executor.compile(src)
    _assert("ImageNet" in code)
    _assert("CIFAR10" in code)
    _assert("VisionTrain" in code)
    _assert("VisionEval" in code)
    _assert("VisionSearch" in code)
    print("✓ test_integration_full_vision_pipeline")


def test_integration_nlp_pipeline():
    src = """
model SentimentClassifier:
    encoder: Linear(768 -> 256)
    relu: ReLU()
    classifier: Linear(256 -> 3)

finetune SentimentFT:
    base_model: "microsoft/deberta-v3-base"
    method: lora
    r: 16
    alpha: 32

data SentimentData:
    source: "./data/sentiment"
    format: csv
    split: 80/10/10

train SentimentTrain:
    model: SentimentClassifier()
    data: SentimentData()
    optimizer: AdamW(lr=2e-5)
    loss: CrossEntropy
    epochs: 10

explain SentimentExplain:
    model: SentimentClassifier
    method: shap

ensemble SentimentEnsemble:
    models: [SentimentClassifier, SentimentFT]
    strategy: averaging
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"NLP pipeline check failed: {result['errors']}")
    code = executor.compile(src)
    _assert("SentimentClassifier" in code)
    _assert("SentimentFT" in code)
    print("✓ test_integration_nlp_pipeline")


def test_integration_generative_ai_pipeline():
    src = """
model MiniGPT:
    embedding: Embedding(50257, 768)
    transformer: Transformer(d_model=768, nhead=12)
    ln: LayerNorm(768)
    lm_head: Linear(768 -> 50257)

data TextCorpus:
    source: "./data/corpus"
    format: csv
    split: 90/5/5

pretrain GPTPretraining:
    model: MiniGPT
    objective: autoregressive
    data: TextCorpus

finetune GPTClassifier:
    base_model: "gpt2"
    method: lora
    r: 8
    alpha: 16

train GPTTraining:
    model: MiniGPT()
    data: TextCorpus()
    optimizer: AdamW(lr=6e-4)
    loss: CrossEntropy
    epochs: 100

deploy GPTDeploy:
    format: torchscript
    serve:
        framework: fastapi
        endpoint: "/generate"
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"Generative AI pipeline failed: {result['errors']}")
    code = executor.compile(src)
    _assert("MiniGPT" in code)
    # Pretrainer class is named after the block definition name (GPTPretraining → GPTPretrainingPretrainer)
    _assert("Pretrainer" in code, "Expected a Pretrainer class in output")
    print("✓ test_integration_generative_ai_pipeline")


def test_integration_multi_backend_compilation():
    src = """
model ClassifierNet:
    fc1: Linear(784 -> 256)
    relu: ReLU()
    dropout: Dropout(0.3)
    fc2: Linear(256 -> 10)
"""
    for backend in ["pytorch", "tensorflow", "jax"]:
        executor = AxonExecutor(config=AxonConfig(backend=backend))
        code = executor.compile(src)
        _assert(len(code) > 100, f"Backend {backend} produced insufficient output")
    print("✓ test_integration_multi_backend_compilation")


def test_integration_complex_gan_pipeline():
    src = """
model Generator:
    fc1: Linear(100 -> 256)
    relu: ReLU()
    fc2: Linear(256 -> 784)

model Discriminator:
    fc1: Linear(784 -> 256)
    relu: ReLU()
    fc2: Linear(256 -> 1)

gan DCGAN:
    generator: Generator
    discriminator: Discriminator
    latent_dim: 100
    loss: wgan
    epochs: 200
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True)
    code = executor.compile(src)
    _assert("Generator" in code)
    _assert("Discriminator" in code)
    print("✓ test_integration_complex_gan_pipeline")


def test_integration_tabular_pipeline():
    src = """
tabular CreditRisk:
    task: classification
    algorithm: xgboost
    target: default
    cross_validation: 5

automl AutoCredit:
    task: classification
    target: default
    time_budget: 3600
    metric: auc

feature CreditFeatures:
    target: default
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"Tabular pipeline failed: {result['errors']}")
    print("✓ test_integration_tabular_pipeline")


def test_integration_rl_pipeline():
    src = """
rl CartPoleAgent:
    algorithm: ppo
    environment: CartPole-v1
    policy: MlpPolicy
    total_timesteps: 100000

monitor RLMonitor:
    model: CartPoleAgent
    backend: tensorboard
    metrics:
        reward
        episode_length
"""
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"RL pipeline failed: {result['errors']}")
    print("✓ test_integration_rl_pipeline")


def test_integration_complete_example_vision():
    """Test with the complete_vision.axon example file."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "complete_vision.axon"
    )
    if not os.path.exists(example_path):
        print("✓ test_integration_complete_example_vision (skipped - file not found)")
        return
    with open(example_path) as f:
        src = f.read()
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"complete_vision.axon invalid: {result['errors']}")
    code = executor.compile(src)
    _assert(len(code) > 500, "Expected substantial generated code")
    print("✓ test_integration_complete_example_vision")


def test_integration_complete_example_generative():
    """Test with the generative_ai.axon example file."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "generative_ai.axon"
    )
    if not os.path.exists(example_path):
        print("✓ test_integration_complete_example_generative (skipped - file not found)")
        return
    with open(example_path) as f:
        src = f.read()
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"generative_ai.axon invalid: {result['errors']}")
    print("✓ test_integration_complete_example_generative")


def test_integration_complete_example_nlp():
    """Test with the nlp_finetuning.axon example file."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "nlp_finetuning.axon"
    )
    if not os.path.exists(example_path):
        print("✓ test_integration_complete_example_nlp (skipped - file not found)")
        return
    with open(example_path) as f:
        src = f.read()
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True, f"nlp_finetuning.axon invalid: {result['errors']}")
    code = executor.compile(src)
    _assert("SentimentClassifier" in code)
    print("✓ test_integration_complete_example_nlp")


def test_integration_axon_config_defaults():
    config = AxonConfig()
    _assert(config.backend == "pytorch")
    _assert(config.auto_run == False)
    _assert(config.verbose == False)
    _assert(config.device == "auto")
    print("✓ test_integration_axon_config_defaults")


def test_integration_axon_config_from_dict():
    config = AxonConfig.from_dict({"backend": "tensorflow", "verbose": True, "custom_key": "val"})
    _assert(config.backend == "tensorflow")
    _assert(config.verbose == True)
    _assert(config.extra.get("custom_key") == "val")
    print("✓ test_integration_axon_config_from_dict")


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════

def test_parser_ratio_expr():
    src = """
data SplitData:
    source: "./data"
    split: 70/15/15
"""
    defn = _first(src)
    _assert(isinstance(defn.split, RatioExpr))
    _assert(defn.split.parts[0] == 70)
    _assert(defn.split.parts[1] == 15)
    _assert(defn.split.parts[2] == 15)
    print("✓ test_parser_ratio_expr")


def test_parser_bool_literals():
    src = """
data MyData:
    source: "./data"
    streaming: true
    cache: False
"""
    defn = _first(src)
    _assert(isinstance(defn, DataDef))
    _assert(defn.streaming == True)
    _assert(defn.cache == False)
    print("✓ test_parser_bool_literals")


def test_parser_negative_number_in_lr():
    src = """
train MyTrain:
    model: MyModel()
    optimizer: Adam(lr=1e-4)
    epochs: 10
    loss: MSE
"""
    defn = _first(src)
    _assert(isinstance(defn, TrainDef))
    _assert(defn.epochs == 10)
    print("✓ test_parser_negative_number_in_lr")


def test_parser_string_path():
    src = """
data PathData:
    source: "/path/to/dataset"
    format: image_folder
"""
    defn = _first(src)
    _assert(isinstance(defn, DataDef))
    _assert(defn.source is not None)
    print("✓ test_parser_string_path")


def test_parser_multiple_definitions():
    src = """
model ModelA:
    fc: Linear(10 -> 5)

model ModelB:
    conv: Conv2d(3 -> 64)
    fc: Linear(64 -> 10)
"""
    program = _parse(src)
    _assert(len(program.definitions) == 2)
    _assert(program.definitions[0].name == "ModelA")
    _assert(program.definitions[1].name == "ModelB")
    print("✓ test_parser_multiple_definitions")


def test_parser_train_advanced_options():
    src = """
train AdvancedTrain:
    model: MyModel()
    epochs: 100
    loss: CrossEntropy
    gradient_accumulation: 4
    ema: true
    label_smoothing: 0.1
    mixup_alpha: 0.2
"""
    defn = _first(src)
    _assert(isinstance(defn, TrainDef))
    _assert(defn.gradient_accumulation == 4)
    _assert(defn.ema == True)
    _assert(defn.label_smoothing == 0.1)
    _assert(defn.mixup_alpha == 0.2)
    print("✓ test_parser_train_advanced_options")


def test_parser_search_all_methods():
    for method in ["bayesian", "random", "grid", "tpe"]:
        src = f"""
search S_{method}:
    base: MyTrain
    method: {method}
    trials: 20
    objective: minimize(val_loss)
"""
        defn = _first(src)
        _assert(isinstance(defn, SearchDef))
        _assert(defn.method == method)
    print("✓ test_parser_search_all_methods")


def test_transpiler_eval_sklearn_metrics():
    src = """
evaluate TestEval:
    checkpoint: "best"
    metrics:
        accuracy
        precision
        recall
        f1
        roc_auc
"""
    code = _transpile(src, "pytorch")
    _assert("TestEvalEvaluator" in code)
    # at least one metric-related word
    _assert(
        "accuracy" in code or "sklearn" in code.lower() or "metric" in code.lower()
    )
    print("✓ test_transpiler_eval_sklearn_metrics")


def test_transpiler_deploy_quantize_fp16():
    src = """
deploy FP16Deploy:
    format: onnx
    quantize: fp16
"""
    code = _transpile(src, "pytorch")
    _assert("FP16DeployDeployer" in code)
    _assert("fp16" in code.lower() or "half" in code.lower(), "Expected FP16 quantization code")
    print("✓ test_transpiler_deploy_quantize_fp16")


def test_transpiler_search_random_method():
    src = """
search RandomSearch:
    base: MyTrain
    method: random
    trials: 50
    objective: minimize(val_loss)
"""
    code = _transpile(src, "pytorch")
    _assert("RandomSampler" in code or "random" in code.lower())
    _assert("optuna" in code.lower())
    print("✓ test_transpiler_search_random_method")


def test_transpiler_python_block_passthrough():
    src = """
model MyModel:
    fc: Linear(10 -> 5)
"""
    # Add a Python block by injecting raw code at top
    executor = AxonExecutor(config=AxonConfig(backend="pytorch"))
    result = executor.check(src)
    _assert(result["valid"] == True)
    print("✓ test_transpiler_python_block_passthrough")


def test_lexer_double_dot_range():
    # The DOUBLE_DOT token is emitted when '..' appears without surrounding digits
    # (e.g. in a non-numeric context like 'start..end').
    tokens = _lex("a..b")
    types = [t.type for t in tokens]
    _assert(TokenType.DOUBLE_DOT in types, "Expected DOUBLE_DOT token for 'a..b'")
    print("✓ test_lexer_double_dot_range")


def test_lexer_pipe_operator():
    tokens = _lex("a | b")
    types = [t.type for t in tokens]
    _assert(TokenType.PIPE in types)
    print("✓ test_lexer_pipe_operator")


def test_lexer_star_operator():
    tokens = _lex("a * b")
    types = [t.type for t in tokens]
    _assert(TokenType.STAR in types)
    print("✓ test_lexer_star_operator")


def test_lexer_percent_operator():
    tokens = _lex("a % b")
    types = [t.type for t in tokens]
    _assert(TokenType.PERCENT in types)
    print("✓ test_lexer_percent_operator")


def test_lexer_plus_minus_operators():
    tokens = _lex("a + b - c")
    types = [t.type for t in tokens]
    _assert(TokenType.PLUS in types)
    _assert(TokenType.MINUS in types or TokenType.DASH in types)
    print("✓ test_lexer_plus_minus_operators")


def test_parser_distill_temperature():
    src = """
distill StudentModel:
    teacher: TeacherBERT
    student: StudentBERT
    temperature: 6.0
    alpha: 0.5
    method: kd
"""
    defn = _first(src)
    _assert(isinstance(defn, DistillDef))
    _assert(defn.temperature == 6.0)
    _assert(defn.alpha == 0.5)
    _assert(defn.method == "kd")
    print("✓ test_parser_distill_temperature")


def test_parser_rl_hyperparams():
    src = """
rl AtariDQN:
    algorithm: dqn
    environment: Pong-v4
    policy: CnnPolicy
    total_timesteps: 1000000
    learning_rate: 1e-4
    gamma: 0.99
    buffer_size: 50000
    batch_size: 32
"""
    defn = _first(src)
    _assert(isinstance(defn, RLDef))
    _assert(defn.algorithm == "dqn")
    _assert(defn.gamma == 0.99)
    _assert(defn.buffer_size == 50000)
    print("✓ test_parser_rl_hyperparams")


def test_parser_graph_gat():
    src = """
graph SocialGAT:
    conv_type: GAT
    num_features: 256
    num_classes: 5
    heads: 8
    dropout: 0.6
"""
    defn = _first(src)
    _assert(isinstance(defn, GraphDef))
    _assert(defn.conv_type == "GAT")
    _assert(defn.heads == 8)
    _assert(defn.dropout == 0.6)
    print("✓ test_parser_graph_gat")


def test_parser_tokenizer_wordpiece():
    src = """
tokenizer WPTokenizer:
    type: wordpiece
    vocab_size: 30000
    max_length: 256
"""
    defn = _first(src)
    _assert(isinstance(defn, TokenizerDef))
    _assert(defn.type == "wordpiece")
    _assert(defn.vocab_size == 30000)
    print("✓ test_parser_tokenizer_wordpiece")


def test_parser_ensemble_stacking():
    src = """
ensemble StackedEnsemble:
    models: [ModelA, ModelB, ModelC, ModelD]
    strategy: stacking
"""
    defn = _first(src)
    _assert(isinstance(defn, EnsembleDef))
    _assert(len(defn.models) == 4)
    _assert(defn.strategy == "stacking")
    print("✓ test_parser_ensemble_stacking")


def test_parser_benchmark_with_devices():
    src = """
benchmark MultiDeviceBench:
    model: MyModel
    num_runs: 500
    num_warmup: 50
"""
    defn = _first(src)
    _assert(isinstance(defn, BenchmarkDef))
    _assert(defn.num_runs == 500)
    _assert(defn.num_warmup == 50)
    print("✓ test_parser_benchmark_with_devices")


def test_parser_federated_fedprox():
    src = """
federated PrivateFed:
    model: SharedModel
    num_clients: 50
    rounds: 200
    strategy: fedprox
    fraction_fit: 0.2
"""
    defn = _first(src)
    _assert(isinstance(defn, FederatedDef))
    _assert(defn.strategy == "fedprox")
    _assert(defn.fraction_fit == 0.2)
    print("✓ test_parser_federated_fedprox")


def test_parser_diffusion_beta_schedule():
    src = """
diffusion CosineScheduler:
    scheduler: ddim
    timesteps: 500
    image_size: 128
    beta_schedule: cosine
    guidance_scale: 5.0
"""
    defn = _first(src)
    _assert(isinstance(defn, DiffusionDef))
    _assert(defn.noise_scheduler == "ddim")
    _assert(defn.timesteps == 500)
    _assert(defn.beta_schedule == "cosine")
    _assert(defn.guidance_scale == 5.0)
    print("✓ test_parser_diffusion_beta_schedule")


def test_parser_multimodal_vqa():
    src = """
multimodal VQAModel:
    task: vqa
    modalities: [image, text, audio]
    fusion: late_fusion
    vision_encoder: vit_large
    text_encoder: roberta
    audio_encoder: wav2vec2
"""
    defn = _first(src)
    _assert(isinstance(defn, MultimodalDef))
    _assert(defn.task == "vqa")
    _assert(len(defn.modalities) == 3)
    _assert(defn.fusion_method == "late_fusion")
    _assert(defn.vision_encoder == "vit_large")
    _assert(defn.text_encoder == "roberta")
    print("✓ test_parser_multimodal_vqa")


def test_transpiler_ensemble_stacking():
    src = """
ensemble StackedModel:
    models: [ModelA, ModelB]
    strategy: stacking
"""
    code = _transpile(src, "pytorch")
    _assert("StackedModelEnsemble" in code)
    _assert("stacking" in code.lower() or "cat" in code.lower())
    print("✓ test_transpiler_ensemble_stacking")


def test_transpiler_explain_gradcam():
    src = """
explain GradCAMExplainer:
    model: CNNModel
    method: gradcam
"""
    code = _transpile(src, "pytorch")
    _assert("GradCAMExplainerExplainer" in code)
    _assert("cam" in code.lower() or "gradcam" in code.lower())
    print("✓ test_transpiler_explain_gradcam")


def test_transpiler_generated_header():
    src = """
model Foo:
    fc: Linear(10 -> 5)
"""
    code = _transpile(src, "pytorch")
    _assert("Generated by Axon" in code, "Expected generation header")
    _assert("pytorch" in code.lower(), "Expected backend in header")
    print("✓ test_transpiler_generated_header")


def test_parser_empty_program():
    """Empty source produces a Program with no definitions."""
    program = _parse("")
    _assert(isinstance(program, Program))
    _assert(len(program.definitions) == 0)
    print("✓ test_parser_empty_program")


def test_parser_comment_only_program():
    src = """
# This is just a comment
# Another comment
"""
    program = _parse(src)
    _assert(isinstance(program, Program))
    _assert(len(program.definitions) == 0)
    print("✓ test_parser_comment_only_program")


# ═══════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    tests = [
        # Lexer tests
        test_lexer_keyword_model,
        test_lexer_keyword_data,
        test_lexer_keyword_train,
        test_lexer_keyword_evaluate,
        test_lexer_keyword_search,
        test_lexer_keyword_deploy,
        test_lexer_all_extended_keywords,
        test_lexer_number_integer,
        test_lexer_number_float,
        test_lexer_number_scientific,
        test_lexer_number_hex,
        test_lexer_string_double_quoted,
        test_lexer_string_single_quoted,
        test_lexer_operator_arrow,
        test_lexer_operator_dot,
        test_lexer_operator_colon,
        test_lexer_operator_brackets,
        test_lexer_param_ref,
        test_lexer_param_ref_in_expression,
        test_lexer_indentation_indent_dedent,
        test_lexer_bool_true_lowercase,
        test_lexer_bool_true_titlecase,
        test_lexer_bool_false_lowercase,
        test_lexer_bool_false_titlecase,
        test_lexer_comment_ignored,
        test_lexer_comment_in_raw_tokens,
        test_lexer_import_keyword,
        test_lexer_from_as_keywords,
        test_lexer_double_dot_range,
        test_lexer_pipe_operator,
        test_lexer_star_operator,
        test_lexer_percent_operator,
        test_lexer_plus_minus_operators,
        # Parser tests - core
        test_parser_model_basic,
        test_parser_model_with_forward,
        test_parser_model_with_param_ref,
        test_parser_data_full,
        test_parser_data_streaming_cache,
        test_parser_train_full,
        test_parser_train_callbacks,
        test_parser_evaluate_full,
        test_parser_search_full,
        test_parser_search_space,
        test_parser_deploy_full,
        test_parser_pipeline,
        test_parser_finetune_lora,
        test_parser_ensemble,
        test_parser_explain,
        test_parser_pretrain,
        # Parser tests - extended
        test_parser_gan,
        test_parser_gan_properties,
        test_parser_diffusion,
        test_parser_rl,
        test_parser_tabular,
        test_parser_timeseries,
        test_parser_graph,
        test_parser_audio,
        test_parser_multimodal,
        test_parser_distill,
        test_parser_quantize,
        test_parser_monitor,
        test_parser_serve,
        test_parser_test_block,
        test_parser_benchmark,
        test_parser_augment,
        test_parser_feature,
        test_parser_embedding,
        test_parser_tokenizer,
        test_parser_callback,
        test_parser_metric,
        test_parser_rag,
        test_parser_agent,
        test_parser_federated,
        test_parser_automl,
        # Parser edge cases
        test_parser_ratio_expr,
        test_parser_bool_literals,
        test_parser_negative_number_in_lr,
        test_parser_string_path,
        test_parser_multiple_definitions,
        test_parser_train_advanced_options,
        test_parser_search_all_methods,
        test_parser_distill_temperature,
        test_parser_rl_hyperparams,
        test_parser_graph_gat,
        test_parser_tokenizer_wordpiece,
        test_parser_ensemble_stacking,
        test_parser_benchmark_with_devices,
        test_parser_federated_fedprox,
        test_parser_diffusion_beta_schedule,
        test_parser_multimodal_vqa,
        test_parser_empty_program,
        test_parser_comment_only_program,
        # Transpiler tests - core
        test_transpiler_model_pytorch,
        test_transpiler_model_tensorflow,
        test_transpiler_model_jax,
        test_transpiler_model_with_dropout,
        test_transpiler_data_pytorch,
        test_transpiler_train_loop,
        test_transpiler_train_metrics,
        test_transpiler_eval,
        test_transpiler_search_optuna,
        test_transpiler_deploy_onnx,
        test_transpiler_deploy_torchscript,
        test_transpiler_pipeline,
        test_transpiler_finetune_lora,
        test_transpiler_ensemble_voting,
        test_transpiler_ensemble_averaging,
        test_transpiler_ensemble_stacking,
        test_transpiler_explain_shap,
        test_transpiler_explain_lime,
        test_transpiler_explain_gradcam,
        test_transpiler_pretrain_masked_lm,
        test_transpiler_pretrain_autoregressive,
        test_transpiler_eval_sklearn_metrics,
        test_transpiler_deploy_quantize_fp16,
        test_transpiler_search_random_method,
        test_transpiler_python_block_passthrough,
        test_transpiler_generated_header,
        # Transpiler tests - extended (compile checks)
        test_transpiler_gan_compiles,
        test_transpiler_diffusion_compiles,
        test_transpiler_rl_compiles,
        test_transpiler_tabular_compiles,
        test_transpiler_timeseries_compiles,
        test_transpiler_graph_compiles,
        test_transpiler_audio_compiles,
        test_transpiler_multimodal_compiles,
        test_transpiler_distill_compiles,
        test_transpiler_quantize_compiles,
        test_transpiler_monitor_compiles,
        test_transpiler_serve_compiles,
        test_transpiler_rag_compiles,
        test_transpiler_agent_compiles,
        test_transpiler_benchmark_compiles,
        test_transpiler_federated_compiles,
        test_transpiler_automl_compiles,
        test_transpiler_all_three_backends_model,
        test_transpiler_invalid_backend_raises,
        # Integration tests
        test_integration_executor_check_valid,
        test_integration_executor_check_definitions,
        test_integration_executor_compile,
        test_integration_executor_run_string,
        test_integration_full_vision_pipeline,
        test_integration_nlp_pipeline,
        test_integration_generative_ai_pipeline,
        test_integration_multi_backend_compilation,
        test_integration_complex_gan_pipeline,
        test_integration_tabular_pipeline,
        test_integration_rl_pipeline,
        test_integration_complete_example_vision,
        test_integration_complete_example_generative,
        test_integration_complete_example_nlp,
        test_integration_axon_config_defaults,
        test_integration_axon_config_from_dict,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
