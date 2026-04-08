"""
Axon v2.0 — A Programming Language for ML/AI Built on Python
==============================================================

Axon is a comprehensive domain-specific language that transpiles to Python,
covering the ENTIRE ML/AI ecosystem:

  Core:        model, data, train, evaluate, search, deploy
  Advanced:    finetune, ensemble, explain, pretrain, pipeline
  Generative:  gan, diffusion
  RL:          rl (PPO, DQN, A2C, SAC, TD3, DDPG)
  Domains:     tabular, timeseries, graph, audio, multimodal
  Optimization: distill, quantize, benchmark
  MLOps:       monitor, serve, test, augment, feature
  NLP:         tokenizer, embedding, rag, agent
  Scale:       federated, automl

Backends: PyTorch, TensorFlow, JAX
Libraries: 50+ Python ML libraries supported
"""

__version__ = "0.1.2"
__author__ = "Axon Language Project"

from axon.transpiler.engine import AxonTranspiler
from axon.parser.lexer import AxonLexer
from axon.parser.parser import AxonParser
from axon.runtime.executor import AxonExecutor
from axon.runtime.config import AxonConfig

__all__ = [
    "AxonTranspiler",
    "AxonLexer",
    "AxonParser",
    "AxonExecutor",
    "AxonConfig",
]
