"""
axon.lsp — Language Server Protocol implementation for the Axon language.

Exports
-------
AxonLanguageServer
    Main LSP server class.  Run with ``AxonLanguageServer().run()``.

CompletionProvider
    ML-aware completion engine.

DiagnosticsProvider
    Real-time diagnostics (parse errors + semantic checks).

HoverProvider
    Hover documentation for keywords, layers, optimizers, etc.
"""

from axon.lsp.server import AxonLanguageServer
from axon.lsp.completions import CompletionProvider
from axon.lsp.diagnostics import DiagnosticsProvider
from axon.lsp.hover import HoverProvider

__all__ = [
    "AxonLanguageServer",
    "CompletionProvider",
    "DiagnosticsProvider",
    "HoverProvider",
]
