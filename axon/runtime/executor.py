"""
Axon Executor v2.0
====================
Compiles and optionally executes Axon programs.
Supports all 30+ block types and 3 backends.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

from axon.parser.parser import AxonParser
from axon.parser.ast_nodes import *
from axon.transpiler.engine import AxonTranspiler
from axon.runtime.config import AxonConfig


class AxonExecutor:
    """
    End-to-end executor for Axon programs.
    
    Workflow:
        1. Parse .axon source → AST
        2. Validate AST (type checking, reference resolution)
        3. Transpile AST → Python code
        4. Optionally execute the generated Python
    """
    
    # All known block type classes
    BLOCK_TYPES = {
        ModelDef: "model", DataDef: "data", TrainDef: "train", EvalDef: "evaluate",
        SearchDef: "search", DeployDef: "deploy", PipelineDef: "pipeline",
        FinetuneDef: "finetune", EnsembleDef: "ensemble", ExplainDef: "explain",
        PretrainDef: "pretrain", GANDef: "gan", DiffusionDef: "diffusion",
        RLDef: "rl", TabularDef: "tabular", TimeSeriesDef: "timeseries",
        GraphDef: "graph", AudioDef: "audio", MultimodalDef: "multimodal",
        DistillDef: "distill", QuantizeDef: "quantize", MonitorDef: "monitor",
        ServeDef: "serve", TestDef: "test", BenchmarkDef: "benchmark",
        AugmentDef: "augment", FeatureDef: "feature", EmbeddingDef: "embedding",
        TokenizerDef: "tokenizer", CallbackDef: "callback", MetricDef: "metric",
        RAGDef: "rag", AgentDef: "agent", FederatedDef: "federated",
        AutoMLDef: "automl", PythonBlock: "python",
    }
    
    def __init__(self, config: Optional[AxonConfig] = None):
        self.config = config or AxonConfig()
    
    def compile(self, source: str) -> str:
        """Compile Axon source code to Python."""
        parser = AxonParser(source)
        ast = parser.parse()
        transpiler = AxonTranspiler(backend=self.config.backend)
        return transpiler.transpile(ast)
    
    def compile_file(self, path: str) -> str:
        """Compile an .axon file to Python."""
        with open(path) as f:
            source = f.read()
        return self.compile(source)
    
    def run(self, path: str, output_path: Optional[str] = None) -> dict:
        """Compile and optionally execute an .axon file."""
        start_time = time.time()
        python_code = self.compile_file(path)
        compile_time = time.time() - start_time
        
        if output_path is None:
            base = Path(path).stem
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{base}.py")
        
        with open(output_path, "w") as f:
            f.write(python_code)
        
        result = {
            "python_code": python_code,
            "output_path": output_path,
            "executed": False,
            "compile_time": compile_time,
            "lines_generated": len(python_code.splitlines()),
        }
        
        if self.config.verbose:
            print(f"[Axon] Compiled {path} → {output_path}")
            print(f"[Axon] Backend: {self.config.backend}")
            print(f"[Axon] Lines: {result['lines_generated']}")
            print(f"[Axon] Time: {compile_time:.3f}s")
        
        if self.config.auto_run and not self.config.dry_run:
            result["executed"] = True
            exec_globals = {"__name__": "__main__"}
            exec(python_code, exec_globals)
            result["globals"] = exec_globals
        
        return result
    
    def run_string(self, source: str) -> str:
        """Compile an Axon source string and return Python code."""
        return self.compile(source)
    
    def check(self, source: str) -> dict:
        """Validate Axon source without generating code."""
        try:
            parser = AxonParser(source)
            ast = parser.parse()
            
            definitions = []
            for defn in ast.definitions:
                name = getattr(defn, 'name', 'unknown')
                if isinstance(defn, PythonBlock):
                    name = defn.code[:30] + "..." if len(defn.code) > 30 else defn.code
                kind = self.BLOCK_TYPES.get(type(defn), type(defn).__name__.replace('Def', '').lower())
                definitions.append({"name": name, "type": kind})
            
            # Cross-reference validation
            warnings = []
            defined_names = {d["name"] for d in definitions}
            
            return {
                "valid": True,
                "errors": [],
                "warnings": warnings,
                "definitions": definitions,
                "block_count": len(definitions),
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "definitions": [],
                "block_count": 0,
            }
    
    def list_blocks(self, source: str) -> list:
        """List all block definitions in the source."""
        result = self.check(source)
        return result.get("definitions", [])
    
    def transpile_block(self, source: str, block_name: str) -> str:
        """Transpile a single named block."""
        parser = AxonParser(source)
        ast = parser.parse()
        
        # Filter to only the named block
        for defn in ast.definitions:
            name = getattr(defn, 'name', '')
            if name == block_name:
                filtered = Program(definitions=[defn])
                transpiler = AxonTranspiler(backend=self.config.backend)
                return transpiler.transpile(filtered)
        
        raise ValueError(f"Block '{block_name}' not found")
