# Axon Language — VS Code Extension

This extension provides rich language support for the [Axon ML Programming Language](https://github.com/axon-lang/axon) inside Visual Studio Code.

## Features

| Feature | Description |
|---------|-------------|
| **Syntax Highlighting** | All 35+ Axon keywords, layer names, optimizers, loss functions, strings, numbers, and comments |
| **Auto-Completions** | Context-aware: block keywords at top level; properties, layers, optimizers, losses inside the right block |
| **Hover Documentation** | Hover over any keyword or layer to see its signature and description |
| **Go-to-Definition** | `F12` jumps to the block that defines a name |
| **Find References** | `Shift+F12` lists every reference to a block name |
| **Real-time Diagnostics** | Parse errors and semantic warnings shown as red/yellow squiggles |
| **Document Formatting** | `Shift+Alt+F` normalises blank lines (full formatter integration coming soon) |

## Requirements

* Python 3.8+
* Axon installed: `pip install axon-lang`
* The `axon` executable must be on your `PATH`

## Installation

### From VS Code Marketplace

Search for **"Axon Language"** in the Extensions panel and click *Install*.

### From source

```bash
cd editors/vscode
npm install
npm run package   # creates axon-language-*.vsix
code --install-extension axon-language-*.vsix
```

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `axon.serverPath` | `"axon"` | Path to the Axon executable |
| `axon.serverArgs` | `["lsp"]` | Arguments passed to the server |
| `axon.trace.server` | `"off"` | LSP traffic tracing (`off` / `messages` / `verbose`) |

## Commands

| Command | Description |
|---------|-------------|
| `Axon: Restart Language Server` | Stops and restarts the LSP server |
| `Axon: Show Output Channel` | Opens the Axon output panel |

## Quick Start

1. Open or create a `.axon` file.
2. The extension activates automatically.
3. Start typing — completions appear as you type.

```axon
model ResNet:
    conv1: Conv2d(3 -> 64, kernel_size=7, stride=2)
    bn1: BatchNorm2d(64)
    relu: ReLU()
    pool: MaxPool2d(kernel_size=3, stride=2)

data ImageNet:
    source: "/data/imagenet"
    format: image_folder
    split: 80/10/10
    loader:
        batch_size: 256
        shuffle: true
        num_workers: 8

train Experiment:
    model: ResNet()
    data: ImageNet()
    optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)
    loss: CrossEntropy
    epochs: 90
    device: auto
    metrics:
        accuracy
        top5_accuracy
```

## Language Server Protocol

The extension launches `axon lsp` as a subprocess and communicates via
JSON-RPC 2.0 over stdin/stdout.  If the server fails to start, run:

```bash
axon lsp  # should block, waiting for LSP messages on stdin
```

and check the error output.

## Changelog

### 0.1.0
- Initial release
- Syntax highlighting for all 35+ keywords, layers, optimizers, losses
- LSP integration (completions, hover, diagnostics, definition, references, formatting)
- TextMate grammar (`axon.tmLanguage.json`)

## License

MIT — see the [Axon repository](https://github.com/axon-lang/axon) for details.
