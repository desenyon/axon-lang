"""
Axon Plugin System
==================
Provides a comprehensive plugin API for extending the Axon language with
new block types, custom parse/transpile logic, completions, and lint rules.

Usage
-----
Create a plugin by subclassing AxonPlugin::

    class MyPlugin(AxonPlugin):
        name = "my-plugin"
        version = "1.0.0"
        block_types = ["myblock"]

        def register(self, registry):
            registry.register_plugin(self)

        def parse_block(self, name, parser):
            # parse tokens and return an ASTNode
            ...

        def transpile_block(self, node, transpiler):
            return "# transpiled myblock"

Then install the package with entry_point group "axon.plugins"::

    [options.entry_points]
    axon.plugins =
        my-plugin = my_package.module:MyPlugin

"""

from __future__ import annotations

import json
import importlib
import importlib.util
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Hook = Callable[..., Any]

# ---------------------------------------------------------------------------
# LintRule
# ---------------------------------------------------------------------------


@dataclass
class LintRule:
    """A single lint rule returned by a plugin."""

    rule_id: str
    description: str
    severity: str = "warning"  # "error" | "warning" | "info"
    check: Optional[Callable[[Any], List[str]]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# PluginContext
# ---------------------------------------------------------------------------


class PluginContext:
    """Sandboxed context object passed to plugins during parse/transpile.

    Exposes only the parts of the parser / transpiler internals that plugins
    are allowed to touch, shielding the rest from accidental mutation.
    """

    def __init__(self, parser_or_transpiler: Any = None):
        self._delegate = parser_or_transpiler

    # --- parser helpers ----

    def peek(self, offset: int = 0):
        """Peek at the next token without consuming it."""
        if hasattr(self._delegate, "peek"):
            return self._delegate.peek(offset)
        return None

    def advance(self):
        """Consume and return the next token."""
        if hasattr(self._delegate, "advance"):
            return self._delegate.advance()
        return None

    def expect(self, token_type):
        """Expect a specific token type and consume it."""
        if hasattr(self._delegate, "expect"):
            return self._delegate.expect(token_type)
        return None

    def match(self, *types):
        """Optionally consume a token of one of the given types."""
        if hasattr(self._delegate, "match"):
            return self._delegate.match(*types)
        return None

    def parse_body(self):
        """Parse an indented block body (key: value pairs)."""
        if hasattr(self._delegate, "_parse_body"):
            return self._delegate._parse_body()
        return {}

    # --- transpiler helpers ----

    def emit(self, code: str) -> str:
        """Return code string (pass-through helper for transpiler plugins)."""
        return code

    def get_indent(self, level: int = 1) -> str:
        return "    " * level


# ---------------------------------------------------------------------------
# AxonPlugin base class
# ---------------------------------------------------------------------------


class AxonPlugin(ABC):
    """Base class for all Axon plugins.

    Subclass this and override the methods you need.  At minimum you must
    set :attr:`name`, :attr:`version`, and :attr:`block_types`.
    """

    #: Human-readable plugin identifier (must be unique)
    name: str = ""
    #: SemVer string, e.g. "1.0.0"
    version: str = "0.1.0"
    #: New block-type keywords this plugin contributes (lowercase)
    block_types: List[str] = []

    def register(self, registry: "PluginRegistry") -> None:
        """Called once when the plugin is loaded.

        The default implementation simply calls ``registry.register_plugin(self)``.
        Override to perform additional setup (e.g. register hooks).
        """
        registry.register_plugin(self)

    def parse_block(self, name: str, parser: Any) -> Any:
        """Parse a block whose keyword matches one of :attr:`block_types`.

        Parameters
        ----------
        name:
            The block-type keyword that triggered this call.
        parser:
            The live :class:`AxonParser` instance.  Use its token-navigation
            methods (``peek``, ``advance``, ``expect``, etc.) to consume
            tokens and return an ``ASTNode`` (or any object the transpiler
            understands).

        Returns
        -------
        ASTNode or None
        """
        return None

    def transpile_block(self, node: Any, transpiler: Any) -> str:
        """Transpile an AST node produced by :meth:`parse_block`.

        Parameters
        ----------
        node:
            The AST node returned by :meth:`parse_block`.
        transpiler:
            The live ``AxonTranspiler`` instance.

        Returns
        -------
        str
            Generated Python source code for this block.
        """
        return ""

    def get_completions(self, context: Any) -> List[str]:
        """Return LSP completion strings relevant to this plugin's block types."""
        return []

    def get_lint_rules(self) -> List[LintRule]:
        """Return lint rules contributed by this plugin."""
        return []


# ---------------------------------------------------------------------------
# HookRegistry helpers
# ---------------------------------------------------------------------------


class _HookStore:
    """Internal storage for named hooks."""

    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {}

    def register(self, event: str, fn: Hook) -> None:
        self._hooks.setdefault(event, []).append(fn)

    def fire(self, event: str, *args, **kwargs) -> List[Any]:
        """Fire all hooks for *event*, return list of results."""
        results = []
        for fn in self._hooks.get(event, []):
            try:
                result = fn(*args, **kwargs)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                # Hook errors must not crash the main pipeline
                results.append(exc)
        return results

    def list_events(self) -> List[str]:
        return list(self._hooks.keys())


# ---------------------------------------------------------------------------
# PluginRegistry (Singleton)
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Central registry for all loaded Axon plugins.

    This class uses the Singleton pattern — every call to ``PluginRegistry()``
    returns the same instance.

    Example::

        registry = PluginRegistry()
        registry.discover_plugins()

        parser_fn = registry.get_block_parser("visualization")
        if parser_fn:
            node = parser_fn("visualization", parser_instance)
    """

    _instance: Optional["PluginRegistry"] = None

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._plugins: Dict[str, AxonPlugin] = {}
            inst._block_parsers: Dict[str, Callable] = {}
            inst._block_transpilers: Dict[str, Callable] = {}
            inst._hooks = _HookStore()
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: AxonPlugin) -> None:
        """Register a plugin instance.

        Iterates over :attr:`~AxonPlugin.block_types` and wires up the
        ``parse_block`` / ``transpile_block`` methods automatically.

        Parameters
        ----------
        plugin:
            An instantiated :class:`AxonPlugin` subclass.
        """
        if not plugin.name:
            raise ValueError("Plugin must have a non-empty 'name'")

        self._plugins[plugin.name] = plugin

        for block_type in plugin.block_types:
            bt = block_type.lower()
            # Capture *plugin* by default-arg binding to avoid late-binding issues
            self._block_parsers[bt] = (
                lambda name, p, _plugin=plugin: _plugin.parse_block(name, p)
            )
            self._block_transpilers[bt] = (
                lambda node, t, _plugin=plugin: _plugin.transpile_block(node, t)
            )

    def unregister_plugin(self, name: str) -> bool:
        """Remove a plugin and all its registered block types.

        Returns ``True`` if the plugin was found and removed.
        """
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        for bt in plugin.block_types:
            self._block_parsers.pop(bt.lower(), None)
            self._block_transpilers.pop(bt.lower(), None)
        return True

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self) -> List[str]:
        """Auto-discover installed plugins via ``entry_points`` (group ``axon.plugins``).

        Returns a list of plugin names that were successfully loaded.
        """
        loaded: List[str] = []
        try:
            from importlib.metadata import entry_points  # Python 3.9+

            eps = entry_points(group="axon.plugins")
        except ImportError:
            try:
                # Python 3.8 fallback
                from importlib.metadata import entry_points as _eps

                all_eps = _eps()
                eps = all_eps.get("axon.plugins", [])
            except ImportError:
                return loaded

        for ep in eps:
            name = self.load_plugin_from_entry_point(ep)
            if name:
                loaded.append(name)
        return loaded

    def load_plugin_from_entry_point(self, ep) -> Optional[str]:
        """Load a single entry_point, instantiate the plugin, and register it.

        Parameters
        ----------
        ep:
            An ``importlib.metadata.EntryPoint`` object.

        Returns
        -------
        str or None
            Plugin name on success, ``None`` on failure.
        """
        try:
            plugin_class = ep.load()
            plugin_instance = plugin_class()
            plugin_instance.register(self)
            return plugin_instance.name
        except Exception:  # noqa: BLE001
            return None

    def load_plugin(self, name: str) -> Optional[AxonPlugin]:
        """Load a plugin by its registered name.

        If the plugin is already loaded this is a no-op and returns the
        existing instance.  To load from a dotted module path (e.g.
        ``"my_package.module:MyPlugin"``), use
        :meth:`load_plugin_from_entry_point` instead.

        Returns the plugin instance or ``None`` if not found.
        """
        return self._plugins.get(name)

    def load_plugin_from_manifest(self, manifest_path: str) -> Optional[AxonPlugin]:
        """Load a plugin described by an ``axon-plugin.json`` manifest file.

        The manifest format::

            {
                "name": "axon-custom-blocks",
                "version": "1.0",
                "block_types": ["custom_block"],
                "entry_point": "my_plugin:MyPlugin"
            }

        Parameters
        ----------
        manifest_path:
            Filesystem path to the JSON manifest.

        Returns
        -------
        AxonPlugin or None
        """
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

        entry_point_str = manifest.get("entry_point", "")
        if ":" not in entry_point_str:
            return None

        module_path, class_name = entry_point_str.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            # Override metadata from manifest
            plugin_instance.name = manifest.get("name", plugin_instance.name)
            plugin_instance.version = manifest.get("version", plugin_instance.version)
            if "block_types" in manifest:
                plugin_instance.block_types = manifest["block_types"]
            plugin_instance.register(self)
            return plugin_instance
        except Exception:  # noqa: BLE001
            return None

    def load_plugin_from_module(
        self, module_path: str, class_name: str
    ) -> Optional[AxonPlugin]:
        """Programmatically load a plugin class from a dotted module path.

        Parameters
        ----------
        module_path:
            Dotted Python module path, e.g. ``"my_package.plugins"``.
        class_name:
            The class inside that module that subclasses :class:`AxonPlugin`.
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            plugin_instance.register(self)
            return plugin_instance
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_block_parser(self, block_type: str) -> Optional[Callable]:
        """Return the parser callable for *block_type*, or ``None``."""
        return self._block_parsers.get(block_type.lower())

    def get_block_transpiler(self, block_type: str) -> Optional[Callable]:
        """Return the transpiler callable for *block_type*, or ``None``."""
        return self._block_transpilers.get(block_type.lower())

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of dicts describing each loaded plugin."""
        result = []
        for plugin in self._plugins.values():
            result.append(
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "block_types": list(plugin.block_types),
                    "lint_rules": [r.rule_id for r in plugin.get_lint_rules()],
                }
            )
        return result

    def get_all_completions(self, context: Any = None) -> List[str]:
        """Aggregate completions from all loaded plugins."""
        completions: List[str] = []
        for plugin in self._plugins.values():
            completions.extend(plugin.get_completions(context))
        return completions

    def get_all_lint_rules(self) -> List[LintRule]:
        """Aggregate lint rules from all loaded plugins."""
        rules: List[LintRule] = []
        for plugin in self._plugins.values():
            rules.extend(plugin.get_lint_rules())
        return rules

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def register_hook(self, event: str, fn: Hook) -> None:
        """Register a hook for a lifecycle *event*.

        Supported events:

        * ``pre_parse``   — called before parsing starts; signature ``(source: str) -> str``
        * ``post_parse``  — called after parsing; signature ``(program) -> program``
        * ``pre_transpile``  — called before transpilation; signature ``(program) -> program``
        * ``post_transpile`` — called after transpilation; signature ``(code: str) -> str``
        """
        self._hooks.register(event, fn)

    def fire_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Fire all registered hooks for *event*."""
        return self._hooks.fire(event, *args, **kwargs)

    # ------------------------------------------------------------------
    # Reset (useful for tests)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton.  Primarily for test isolation."""
        cls._instance = None

    def clear(self) -> None:
        """Remove all registered plugins and hooks without destroying the singleton."""
        self._plugins.clear()
        self._block_parsers.clear()
        self._block_transpilers.clear()
        self._hooks = _HookStore()
