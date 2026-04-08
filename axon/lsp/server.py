"""
Axon Language Server Protocol (LSP) Server
============================================
Implements JSON-RPC 2.0 over stdin/stdout (the standard LSP transport).

Supported LSP methods
---------------------
* initialize / initialized
* textDocument/didOpen
* textDocument/didChange
* textDocument/didSave
* textDocument/completion
* textDocument/hover
* textDocument/definition
* textDocument/references
* textDocument/formatting
* textDocument/publishDiagnostics  (push, not request)
* shutdown / exit

Usage::

    python -m axon lsp
    # or
    axon lsp
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

from axon.lsp.completions import CompletionProvider
from axon.lsp.diagnostics import DiagnosticsProvider
from axon.lsp.hover import HoverProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _read_message(stream) -> Optional[Dict[str, Any]]:
    """Read one JSON-RPC message from *stream* (stdin).

    Messages follow the LSP framing::

        Content-Length: <N>\\r\\n
        \\r\\n
        <JSON body of N bytes>
    """
    headers: Dict[str, str] = {}

    while True:
        raw = stream.readline()
        if not raw:
            return None  # EOF
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            break  # blank line separates headers from body
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", 0))
    if content_length == 0:
        return None

    body = stream.read(content_length)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _write_message(stream, message: Dict[str, Any]) -> None:
    """Write one JSON-RPC message to *stream* (stdout) with LSP framing."""
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header + body)
    stream.flush()


def _make_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _make_error_response(
    request_id: Any, code: int, message: str
) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _make_notification(method: str, params: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params}


# ---------------------------------------------------------------------------
# Server capabilities
# ---------------------------------------------------------------------------

SERVER_CAPABILITIES = {
    "textDocumentSync": {
        "openClose": True,
        "change": 1,  # Full sync
        "save": {"includeText": True},
    },
    "completionProvider": {
        "triggerCharacters": [":", " ", "\n"],
        "resolveProvider": False,
    },
    "hoverProvider": True,
    "definitionProvider": True,
    "referencesProvider": True,
    "documentFormattingProvider": True,
}


# ---------------------------------------------------------------------------
# AxonLanguageServer
# ---------------------------------------------------------------------------


class AxonLanguageServer:
    """Full LSP server for the Axon language.

    Transport
    ---------
    JSON-RPC 2.0 over stdin/stdout following the Language Server Protocol
    specification (LSP 3.17).

    Usage::

        server = AxonLanguageServer()
        server.run()
    """

    def __init__(
        self,
        in_stream=None,
        out_stream=None,
    ):
        self._in = in_stream or sys.stdin.buffer
        self._out = out_stream or sys.stdout.buffer
        self._documents: Dict[str, str] = {}  # uri -> source text
        self._shutdown = False
        self._initialized = False

        self._completion_provider = CompletionProvider()
        self._diagnostics_provider = DiagnosticsProvider()
        self._hover_provider = HoverProvider()

        # Dispatch table: method -> handler
        self._handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "exit": self._handle_exit,
            "textDocument/didOpen": self._handle_did_open,
            "textDocument/didChange": self._handle_did_change,
            "textDocument/didSave": self._handle_did_save,
            "textDocument/didClose": self._handle_did_close,
            "textDocument/completion": self._handle_completion,
            "textDocument/hover": self._handle_hover,
            "textDocument/definition": self._handle_definition,
            "textDocument/references": self._handle_references,
            "textDocument/formatting": self._handle_formatting,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the server's message loop (blocks until exit)."""
        logger.info("Axon LSP server starting (stdin/stdout transport)")
        while not self._shutdown:
            msg = _read_message(self._in)
            if msg is None:
                break
            self._dispatch(msg)

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        handler = self._handlers.get(method)
        if handler is None:
            if msg_id is not None:
                self._send(_make_error_response(msg_id, -32601, f"Method not found: {method}"))
            return

        try:
            result = handler(params)
            if msg_id is not None:
                self._send(_make_response(msg_id, result))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error handling %s", method)
            if msg_id is not None:
                self._send(_make_error_response(msg_id, -32603, str(exc)))

    def _send(self, message: Dict[str, Any]) -> None:
        _write_message(self._out, message)

    def _notify(self, method: str, params: Any) -> None:
        self._send(_make_notification(method, params))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._initialized = True
        return {
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": {
                "name": "axon-language-server",
                "version": self._get_version(),
            },
        }

    def _handle_initialized(self, params: Dict[str, Any]) -> None:
        logger.info("Client initialized")
        return None

    def _handle_shutdown(self, params: Dict[str, Any]) -> None:
        self._shutdown = True
        return None

    def _handle_exit(self, params: Dict[str, Any]) -> None:
        sys.exit(0)

    # --- text document lifecycle ---

    def _handle_did_open(self, params: Dict[str, Any]) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri", "")
        text = doc.get("text", "")
        self._documents[uri] = text
        self._publish_diagnostics(uri, text)
        return None

    def _handle_did_change(self, params: Dict[str, Any]) -> None:
        uri = params.get("textDocument", {}).get("uri", "")
        changes = params.get("contentChanges", [])
        if changes:
            # We use full-sync (change type 1) so the last change is the full text
            text = changes[-1].get("text", "")
            self._documents[uri] = text
            self._publish_diagnostics(uri, text)
        return None

    def _handle_did_save(self, params: Dict[str, Any]) -> None:
        uri = params.get("textDocument", {}).get("uri", "")
        text = params.get("text")
        if text is not None:
            self._documents[uri] = text
        source = self._documents.get(uri, "")
        self._publish_diagnostics(uri, source)
        return None

    def _handle_did_close(self, params: Dict[str, Any]) -> None:
        uri = params.get("textDocument", {}).get("uri", "")
        self._documents.pop(uri, None)
        # Clear diagnostics
        self._notify(
            "textDocument/publishDiagnostics",
            {"uri": uri, "diagnostics": []},
        )
        return None

    # --- features ---

    def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("textDocument", {}).get("uri", "")
        position = params.get("position", {})
        line = position.get("line", 0)
        character = position.get("character", 0)

        source = self._documents.get(uri, "")
        items = self._completion_provider.get_completions(source, line, character, uri)

        return {"isIncomplete": False, "items": items}

    def _handle_hover(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        uri = params.get("textDocument", {}).get("uri", "")
        position = params.get("position", {})
        line = position.get("line", 0)
        character = position.get("character", 0)

        source = self._documents.get(uri, "")
        return self._hover_provider.get_hover(source, line, character, uri)

    def _handle_definition(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Go-to-definition: find where a block is defined."""
        uri = params.get("textDocument", {}).get("uri", "")
        position = params.get("position", {})
        line = position.get("line", 0)
        character = position.get("character", 0)

        source = self._documents.get(uri, "")
        word = self._hover_provider._word_at(source, line, character)
        if not word:
            return None

        locations = self._find_block_definitions(source, word, uri)
        return locations or None

    def _handle_references(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all references to a block name."""
        uri = params.get("textDocument", {}).get("uri", "")
        position = params.get("position", {})
        line = position.get("line", 0)
        character = position.get("character", 0)

        source = self._documents.get(uri, "")
        word = self._hover_provider._word_at(source, line, character)
        if not word:
            return []

        return self._find_references(source, word, uri)

    def _handle_formatting(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Format the entire document using the Axon formatter (if available)."""
        uri = params.get("textDocument", {}).get("uri", "")
        source = self._documents.get(uri, "")

        formatted = self._format_source(source)
        if formatted is None or formatted == source:
            return None

        lines = source.splitlines()
        end_line = len(lines)
        end_char = len(lines[-1]) if lines else 0

        return [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": end_line, "character": end_char},
                },
                "newText": formatted,
            }
        ]

    # ------------------------------------------------------------------
    # Diagnostics push
    # ------------------------------------------------------------------

    def _publish_diagnostics(self, uri: str, source: str) -> None:
        diags = self._diagnostics_provider.get_diagnostics(source, uri)
        self._notify(
            "textDocument/publishDiagnostics",
            {"uri": uri, "diagnostics": diags},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_block_definitions(
        self, source: str, name: str, uri: str
    ) -> List[Dict[str, Any]]:
        """Locate all block definitions that declare *name*."""
        import re

        locations = []
        pattern = re.compile(
            r"^\s*(?:model|data|train|evaluate|search|deploy|pipeline|finetune|ensemble"
            r"|explain|pretrain|gan|diffusion|rl|tabular|timeseries|graph|audio"
            r"|multimodal|distill|quantize|monitor|serve|test|benchmark|augment"
            r"|feature|embedding|tokenizer|callback|metric|rag|agent|federated|automl"
            r")\s+" + re.escape(name) + r"\s*:",
            re.MULTILINE,
        )
        for m in pattern.finditer(source):
            line_no = source[: m.start()].count("\n")
            col = m.start() - source[: m.start()].rfind("\n") - 1
            locations.append(
                {
                    "uri": uri,
                    "range": {
                        "start": {"line": line_no, "character": col},
                        "end": {"line": line_no, "character": col + len(m.group())},
                    },
                }
            )
        return locations

    def _find_references(
        self, source: str, name: str, uri: str
    ) -> List[Dict[str, Any]]:
        """Find all occurrences of *name* as an identifier in *source*."""
        import re

        refs = []
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        lines = source.splitlines()
        for line_no, line_text in enumerate(lines):
            for m in pattern.finditer(line_text):
                refs.append(
                    {
                        "uri": uri,
                        "range": {
                            "start": {"line": line_no, "character": m.start()},
                            "end": {"line": line_no, "character": m.end()},
                        },
                    }
                )
        return refs

    @staticmethod
    def _format_source(source: str) -> Optional[str]:
        """Attempt to format *source* using the Axon formatter."""
        try:
            from axon.formatter import AxonFormatter  # type: ignore[import]

            formatter = AxonFormatter()
            return formatter.format(source)
        except ImportError:
            pass
        # Basic fallback: normalise blank lines
        import re

        formatted = re.sub(r"\n{3,}", "\n\n", source.strip()) + "\n"
        return formatted

    @staticmethod
    def _get_version() -> str:
        try:
            from axon import __version__

            return __version__
        except ImportError:
            return "0.0.0"
