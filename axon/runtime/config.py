"""
Axon Configuration v2.0
========================
Comprehensive configuration for the Axon runtime.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AxonConfig:
    """Configuration for the Axon transpiler and runtime."""
    
    # Backend selection
    backend: str = "pytorch"  # pytorch | tensorflow | jax
    
    # Output settings
    output_dir: str = "./axon_output"
    output_file: Optional[str] = None
    
    # Execution settings
    auto_run: bool = False
    dry_run: bool = False
    verbose: bool = False
    
    # Device settings
    device: str = "auto"
    precision: str = "fp32"  # fp32 | fp16 | bf16 | mixed
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    tracking_backend: Optional[str] = None  # wandb | mlflow | tensorboard | neptune | comet
    project_name: Optional[str] = None
    
    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False
    
    # Distributed training
    distributed: Optional[str] = None  # ddp | fsdp | deepspeed
    num_gpus: int = 1
    num_nodes: int = 1
    
    # Compilation
    compile_model: bool = False  # torch.compile
    compile_backend: str = "inductor"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "rich"  # rich | plain | json
    
    # Deployment
    target_device: str = "gpu"  # gpu | cpu | edge | mobile | browser
    quantize: Optional[str] = None  # int8 | int4 | fp16 | bf16
    
    # Extra settings
    extra: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: dict) -> "AxonConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        known_args = {k: v for k, v in d.items() if k in known}
        extra_args = {k: v for k, v in d.items() if k not in known}
        config = cls(**known_args)
        config.extra = extra_args
        return config
    
    @classmethod
    def from_file(cls, path: str) -> "AxonConfig":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
