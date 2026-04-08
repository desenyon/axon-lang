# Axon Language Specification v2.0

## Overview
Axon is a domain-specific language built on Python for the complete ML/AI lifecycle. It transpiles to standard Python using PyTorch, TensorFlow, JAX, and 50+ ML libraries.

## Design Principles
1. ML-First Syntax — every keyword maps to an ML concept
2. Zero Boilerplate — common patterns are single declarations
3. Backend Agnostic — write once, target PyTorch/TF/JAX
4. Shape-Aware Tensors — type annotations at the language level
5. Experiment-Native — built-in tracking, search, reproducibility
6. Full Python Interop — escape to raw Python with @python blocks

## File Extension: `.axon`

## 35 Block Types

### Core
`model` `data` `train` `evaluate` `search` `deploy`

### Advanced Training
`finetune` `pretrain` `distill` `ensemble`

### Generative
`gan` `diffusion` `rag` `agent`

### Domain-Specific
`tabular` `timeseries` `graph` `audio` `multimodal` `rl`

### MLOps
`monitor` `serve` `quantize` `benchmark` `test`

### Data & Features  
`augment` `feature` `tokenizer` `embedding`

### Infrastructure
`pipeline` `callback` `metric` `federated` `automl` `explain`

## Syntax Reference

### Blocks
```axon
keyword Name:
    key: value
    key: value
```

### Layer Arrow Notation
```axon
Linear(784 -> 256)  # in_features=784, out_features=256
```

### Parameter References
```axon
head: Linear(2048 -> @num_classes)  # configurable parameter
```

### Split Ratios
```axon
split: 80/10/10  # train/val/test
```

### Function Calls
```axon
optimizer: AdamW(lr=3e-4, weight_decay=1e-4)
```

### Lists
```axon
models: [ModelA, ModelB, ModelC]
```

### Booleans
```axon
shuffle: true
pretrained: false
```

### Comments
```axon
# This is a comment
```

### Python Escape
```axon
@python: import custom_module
```

## Training Advanced Options
```axon
train Name:
    # ... basic options ...
    gradient_accumulation: 4
    gradient_clip_norm: 1.0
    ema: true
    ema_decay: 0.9999
    swa: true
    label_smoothing: 0.1
    mixup_alpha: 0.2
    cutmix_alpha: 1.0
    compile: true
    distributed: fsdp
    num_gpus: 8
    precision: bf16
```

## Keywords (40+)
model, data, train, evaluate, search, deploy, pipeline, transform, pretrain, finetune, ensemble, explain, forward, gan, diffusion, rl, tabular, timeseries, graph, audio, multimodal, distill, quantize, monitor, serve, test, benchmark, augment, feature, embedding, tokenizer, callback, metric, rag, agent, federated, automl, import, from, as, true, false, none, self, return
