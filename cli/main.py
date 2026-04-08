#!/usr/bin/env python3
"""
Axon CLI
=========
Command-line interface for the Axon programming language.

Usage:
    axon compile model.axon                    # Compile to Python
    axon compile model.axon -o model.py        # Compile with output path
    axon compile model.axon --backend jax      # Compile for JAX
    axon run model.axon                        # Compile and execute
    axon check model.axon                      # Validate syntax
    axon repl                                  # Interactive REPL
    axon init project_name                     # Initialize a new project
    axon version                               # Show version
"""

import argparse
import sys
import os

# Add parent dir to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axon.runtime.executor import AxonExecutor
from axon.runtime.config import AxonConfig
from axon import __version__


BANNER = r"""
     _                    
    / \   __  _____  _ __  
   / _ \  \ \/ / _ \| '_ \ 
  / ___ \  >  < (_) | | | |
 /_/   \_\/_/\_\___/|_| |_|
                            
  ML Programming Language v{}
  Backend: {{backend}}
""".format(__version__)


def _check_file_exists(path):
    """Validate that the input file exists."""
    if not os.path.isfile(path):
        print(f"✗ Error: file not found: {path}")
        sys.exit(1)
    if not path.endswith(".axon"):
        print(f"✗ Warning: {path} does not have an .axon extension")


def cmd_compile(args):
    """Compile .axon file to Python."""
    _check_file_exists(args.file)
    config = AxonConfig(
        backend=args.backend,
        output_dir=args.output_dir or "./axon_output",
        verbose=args.verbose,
    )
    executor = AxonExecutor(config=config)
    
    output = args.output
    if not output:
        base = os.path.splitext(os.path.basename(args.file))[0]
        output = os.path.join(config.output_dir, f"{base}.py")
    
    try:
        python_code = executor.compile_file(args.file)
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(python_code)
    
    lines = len(python_code.splitlines())
    print(f"✓ Compiled {args.file} → {output} ({lines} lines)")
    print(f"  Backend: {args.backend}")
    
    if args.verbose:
        print("\n── Generated Python ──")
        print(python_code)


def cmd_run(args):
    """Compile and execute an .axon file."""
    _check_file_exists(args.file)
    config = AxonConfig(
        backend=args.backend,
        output_dir=args.output_dir or "./axon_output",
        auto_run=True,
        verbose=args.verbose,
    )
    executor = AxonExecutor(config=config)
    
    print(BANNER.replace("{backend}", args.backend))
    result = executor.run(args.file)
    print(f"\n✓ Executed {args.file}")


def cmd_check(args):
    """Validate .axon file syntax."""
    _check_file_exists(args.file)
    with open(args.file) as f:
        source = f.read()
    
    executor = AxonExecutor()
    result = executor.check(source)
    
    if result["valid"]:
        print(f"✓ {args.file} is valid")
        for defn in result["definitions"]:
            print(f"  {defn['type']:>10}: {defn['name']}")
    else:
        print(f"✗ {args.file} has errors:")
        for err in result["errors"]:
            print(f"  ERROR: {err}")
    
    return 0 if result["valid"] else 1


def cmd_repl(args):
    """Interactive Axon REPL."""
    config = AxonConfig(backend=args.backend, verbose=True)
    executor = AxonExecutor(config=config)
    
    print(BANNER.replace("{backend}", args.backend))
    print("Type Axon code, then enter a blank line to compile.")
    print("Commands: :quit, :backend <name>, :help\n")
    
    buffer = []
    
    while True:
        try:
            prompt = "axon> " if not buffer else "  ... "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if line.strip() == ":quit":
            print("Goodbye!")
            break
        elif line.strip().startswith(":backend"):
            parts = line.strip().split()
            if len(parts) == 2:
                config.backend = parts[1]
                executor = AxonExecutor(config=config)
                print(f"Backend switched to: {config.backend}")
            continue
        elif line.strip() == ":help":
            print("Commands:")
            print("  :quit           Exit the REPL")
            print("  :backend <name> Switch backend (pytorch/tensorflow/jax)")
            print("  :help           Show this help")
            print("\nEnter Axon code, then press Enter on a blank line to compile.")
            continue
        
        if line.strip() == "" and buffer:
            source = "\n".join(buffer)
            try:
                python_code = executor.run_string(source)
                print("\n── Generated Python ──")
                print(python_code)
                print("── End ──\n")
            except Exception as e:
                print(f"\nError: {e}\n")
            buffer = []
        else:
            buffer.append(line)


def cmd_init(args):
    """Initialize a new Axon project."""
    project_name = args.name
    
    os.makedirs(project_name, exist_ok=True)
    os.makedirs(f"{project_name}/src", exist_ok=True)
    os.makedirs(f"{project_name}/data", exist_ok=True)
    os.makedirs(f"{project_name}/models", exist_ok=True)
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    # axon.config.json
    config = {
        "backend": "pytorch",
        "output_dir": "./generated",
        "device": "auto",
        "seed": 42
    }
    import json
    with open(f"{project_name}/axon.config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Sample .axon file
    sample = '''# My Axon ML Project
# Compile: axon compile src/main.axon
# Run: axon run src/main.axon

model SimpleClassifier:
    features: Linear(784 -> 256)
    relu: ReLU()
    output: Linear(256 -> 10)

data MNISTData:
    source: "./data"
    format: image_folder
    split: 80/10/10
    
    transform:
        resize(28, 28)
        normalize(mean=0.1307, std=0.3081)
    
    loader:
        batch_size: 64
        shuffle: true
        num_workers: 2

train Experiment:
    model: SimpleClassifier()
    data: MNISTData()
    
    optimizer: Adam(lr=1e-3)
    loss: CrossEntropy
    epochs: 10
    device: auto
    
    metrics:
        accuracy
        f1

evaluate Experiment:
    checkpoint: "best"
    data: MNISTData.test
    
    metrics:
        accuracy
        precision
        recall
        confusion_matrix
'''
    
    with open(f"{project_name}/src/main.axon", "w") as f:
        f.write(sample)
    
    # .gitignore
    with open(f"{project_name}/.gitignore", "w") as f:
        f.write("generated/\naxon_output/\n__pycache__/\n*.pyc\nresults/\n*.pth\n*.onnx\n")
    
    # README
    with open(f"{project_name}/README.md", "w") as f:
        f.write(f"# {project_name}\n\n")
        f.write("An ML project built with Axon.\n\n")
        f.write("## Quick Start\n\n```bash\n")
        f.write("axon compile src/main.axon\n")
        f.write("axon run src/main.axon\n")
        f.write("```\n")
    
    print(f"✓ Created Axon project: {project_name}/")
    print(f"  {project_name}/src/main.axon     - Main Axon file")
    print(f"  {project_name}/axon.config.json  - Configuration")
    print(f"  {project_name}/data/             - Dataset directory")
    print(f"  {project_name}/models/           - Saved models")
    print(f"  {project_name}/results/          - Outputs & reports")


def main():
    parser = argparse.ArgumentParser(
        prog="axon",
        description="Axon - ML Programming Language built on Python"
    )
    parser.add_argument("--version", action="version", version=f"Axon {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # compile
    p_compile = subparsers.add_parser("compile", help="Compile .axon to Python")
    p_compile.add_argument("file", help="Path to .axon file")
    p_compile.add_argument("-o", "--output", help="Output Python file path")
    p_compile.add_argument("-b", "--backend", default="pytorch",
                          choices=["pytorch", "tensorflow", "jax"],
                          help="Target backend (default: pytorch)")
    p_compile.add_argument("--output-dir", help="Output directory")
    p_compile.add_argument("-v", "--verbose", action="store_true")
    
    # run
    p_run = subparsers.add_parser("run", help="Compile and execute .axon file")
    p_run.add_argument("file", help="Path to .axon file")
    p_run.add_argument("-b", "--backend", default="pytorch",
                      choices=["pytorch", "tensorflow", "jax"])
    p_run.add_argument("--output-dir", help="Output directory")
    p_run.add_argument("-v", "--verbose", action="store_true")
    
    # check
    p_check = subparsers.add_parser("check", help="Validate .axon syntax")
    p_check.add_argument("file", help="Path to .axon file")
    
    # repl
    p_repl = subparsers.add_parser("repl", help="Interactive REPL")
    p_repl.add_argument("-b", "--backend", default="pytorch",
                       choices=["pytorch", "tensorflow", "jax"])
    
    # init
    p_init = subparsers.add_parser("init", help="Initialize a new Axon project")
    p_init.add_argument("name", help="Project name")
    
    args = parser.parse_args()
    
    if args.command == "compile":
        cmd_compile(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "repl":
        cmd_repl(args)
    elif args.command == "init":
        cmd_init(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
