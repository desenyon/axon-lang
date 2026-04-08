"""
Axon LSP — Diagnostics Provider
================================
Converts parser errors and semantic issues into LSP Diagnostic objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# LSP severity constants
# ---------------------------------------------------------------------------

SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFO = 3
SEVERITY_HINT = 4


def _make_diagnostic(
    start_line: int,
    start_char: int,
    end_line: int,
    end_char: int,
    message: str,
    severity: int = SEVERITY_ERROR,
    source: str = "axon",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single LSP Diagnostic dict."""
    diag: Dict[str, Any] = {
        "range": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
        "severity": severity,
        "source": source,
        "message": message,
    }
    if code is not None:
        diag["code"] = code
    return diag


# ---------------------------------------------------------------------------
# DiagnosticsProvider
# ---------------------------------------------------------------------------


class DiagnosticsProvider:
    """Runs the Axon parser and semantic checks, then returns LSP diagnostics."""

    def __init__(self) -> None:
        self._semantic_rules = self._default_semantic_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_diagnostics(self, source: str, uri: str = "") -> List[Dict[str, Any]]:
        """Parse *source* and return a list of LSP Diagnostic dicts.

        Parameters
        ----------
        source:
            Full document source text.
        uri:
            Document URI (used only for reporting, not for I/O).

        Returns
        -------
        list[dict]
            Zero or more LSP Diagnostic objects.
        """
        diagnostics: List[Dict[str, Any]] = []

        # --- syntax errors ---
        program, parse_diags = self._parse_source(source)
        diagnostics.extend(parse_diags)

        # --- semantic checks ---
        if program is not None:
            diagnostics.extend(self._run_semantic_checks(program, source))

        # --- plugin lint rules ---
        diagnostics.extend(self._run_plugin_rules(program, source))

        return diagnostics

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_source(
        self, source: str
    ) -> tuple:
        """Try to parse *source*, returning (program | None, diagnostics)."""
        try:
            from axon.parser.parser import AxonParser, ParseError

            parser = AxonParser(source)
            program = parser.parse()
            return program, []
        except Exception as exc:
            diag = self._exc_to_diagnostic(exc, source)
            return None, [diag] if diag else []

    def _exc_to_diagnostic(
        self, exc: Exception, source: str
    ) -> Optional[Dict[str, Any]]:
        """Convert a ParseError (or any Exception) to an LSP diagnostic."""
        msg = str(exc)
        # Try to extract line/col from ParseError
        start_line = 0
        start_char = 0

        try:
            from axon.parser.parser import ParseError

            if isinstance(exc, ParseError) and exc.token is not None:
                start_line = max(0, exc.token.line - 1)  # LSP is 0-based
                start_char = max(0, exc.token.col - 1)
        except ImportError:
            pass

        # Fallback: try to parse "line X:Y" from message
        if start_line == 0:
            import re

            m = re.search(r"line\s+(\d+):(\d+)", msg)
            if m:
                start_line = max(0, int(m.group(1)) - 1)
                start_char = max(0, int(m.group(2)) - 1)

        return _make_diagnostic(
            start_line,
            start_char,
            start_line,
            start_char + 10,
            msg,
            SEVERITY_ERROR,
            code="AXN001",
        )

    # ------------------------------------------------------------------
    # Semantic checks
    # ------------------------------------------------------------------

    def _run_semantic_checks(
        self, program: Any, source: str
    ) -> List[Dict[str, Any]]:
        """Run all registered semantic rules against the parsed program."""
        diagnostics: List[Dict[str, Any]] = []
        for rule in self._semantic_rules:
            try:
                diagnostics.extend(rule(program, source))
            except Exception:  # noqa: BLE001
                pass
        return diagnostics

    def _default_semantic_rules(self):
        """Return a list of built-in semantic check functions."""
        return [
            self._check_duplicate_names,
            self._check_train_references,
            self._check_empty_blocks,
        ]

    def _check_duplicate_names(
        self, program: Any, source: str
    ) -> List[Dict[str, Any]]:
        """Warn about block definitions that share the same name."""
        seen: Dict[str, int] = {}  # name → first occurrence line
        diagnostics: List[Dict[str, Any]] = []
        for defn in getattr(program, "definitions", []):
            name = getattr(defn, "name", None)
            line = getattr(defn, "line", 1)
            if name is None:
                continue
            if name in seen:
                diagnostics.append(
                    _make_diagnostic(
                        max(0, line - 1),
                        0,
                        max(0, line - 1),
                        len(name),
                        f"Duplicate block name '{name}' (first defined at line {seen[name]})",
                        SEVERITY_WARNING,
                        code="AXN010",
                    )
                )
            else:
                seen[name] = line
        return diagnostics

    def _check_train_references(
        self, program: Any, source: str
    ) -> List[Dict[str, Any]]:
        """Check that model/data references inside train blocks are defined."""
        diagnostics: List[Dict[str, Any]] = []
        definitions = getattr(program, "definitions", [])

        # Collect all defined block names
        defined_names = set()
        for defn in definitions:
            name = getattr(defn, "name", None)
            if name:
                defined_names.add(name)

        # Check train blocks
        for defn in definitions:
            node_type = type(defn).__name__
            if node_type not in ("TrainDef", "FinetuneDef", "EvaluateDef"):
                continue
            body = getattr(defn, "body", {})
            line = getattr(defn, "line", 1)

            for ref_key in ("model", "data"):
                ref_val = body.get(ref_key)
                if ref_val is None:
                    continue
                # ref_val might be a string or an ASTNode
                ref_str = (
                    getattr(ref_val, "value", None)
                    or (ref_val if isinstance(ref_val, str) else None)
                )
                if ref_str and ref_str not in defined_names:
                    # Strip "()" for calls like "MyModel()"
                    base = ref_str.split("(")[0].strip()
                    if base and base not in defined_names:
                        diagnostics.append(
                            _make_diagnostic(
                                max(0, line - 1),
                                0,
                                max(0, line - 1),
                                20,
                                f"Reference '{base}' in {node_type} is not defined in this file",
                                SEVERITY_WARNING,
                                code="AXN011",
                            )
                        )
        return diagnostics

    def _check_empty_blocks(
        self, program: Any, source: str
    ) -> List[Dict[str, Any]]:
        """Hint about blocks with empty bodies."""
        diagnostics: List[Dict[str, Any]] = []
        for defn in getattr(program, "definitions", []):
            body = getattr(defn, "body", None)
            line = getattr(defn, "line", 1)
            if body is not None and len(body) == 0:
                diagnostics.append(
                    _make_diagnostic(
                        max(0, line - 1),
                        0,
                        max(0, line - 1),
                        20,
                        f"Block '{getattr(defn, 'name', '?')}' has an empty body",
                        SEVERITY_HINT,
                        code="AXN020",
                    )
                )
        return diagnostics

    # ------------------------------------------------------------------
    # Plugin rules
    # ------------------------------------------------------------------

    def _run_plugin_rules(
        self, program: Optional[Any], source: str
    ) -> List[Dict[str, Any]]:
        """Execute lint rules contributed by loaded plugins."""
        if program is None:
            return []
        diagnostics: List[Dict[str, Any]] = []
        try:
            from axon.plugins import PluginRegistry

            registry = PluginRegistry()
            for rule in registry.get_all_lint_rules():
                if rule.check is None:
                    continue
                for defn in getattr(program, "definitions", []):
                    messages = rule.check(defn)
                    if not messages:
                        continue
                    line = max(0, getattr(defn, "line", 1) - 1)
                    for msg in messages:
                        diagnostics.append(
                            _make_diagnostic(
                                line, 0, line, 20, msg, SEVERITY_WARNING, code=rule.rule_id
                            )
                        )
        except Exception:  # noqa: BLE001
            pass
        return diagnostics
