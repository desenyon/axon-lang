"""
Axon Module System
==================
Provides ModuleResolver and ModuleRegistry for handling Axon's import system.

Supports:
  import Model from "./other.axon"          (named import)
  import { Model, DataLoader } from "..."   (destructured import)
  import * from "./all.axon"               (wildcard import)
"""

import os
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass, field


class ModuleError(Exception):
    """Raised when module resolution or loading fails."""
    pass


class CircularImportError(ModuleError):
    """Raised when a circular import is detected."""
    pass


@dataclass
class ModuleInfo:
    """Information about a resolved and parsed module."""
    path: str               # Absolute resolved path
    source: str             # Raw source code
    definitions: dict = field(default_factory=dict)  # name -> AST node
    exports: list = field(default_factory=list)       # exported names
    loaded: bool = False


class ModuleRegistry:
    """Tracks all exported definitions across modules."""

    def __init__(self):
        # Maps absolute path -> ModuleInfo
        self._modules: Dict[str, ModuleInfo] = {}
        # Maps (path, name) -> AST node
        self._exports: Dict[tuple, Any] = {}

    def register_module(self, info: ModuleInfo):
        """Register a loaded module."""
        self._modules[info.path] = info
        for name, node in info.definitions.items():
            self._exports[(info.path, name)] = node

    def get_module(self, path: str) -> Optional[ModuleInfo]:
        return self._modules.get(path)

    def get_export(self, path: str, name: str) -> Optional[Any]:
        return self._exports.get((path, name))

    def get_all_exports(self, path: str) -> dict:
        """Return all exports from a module as {name: node}."""
        info = self._modules.get(path)
        if info is None:
            return {}
        return dict(info.definitions)

    def list_modules(self) -> List[str]:
        return list(self._modules.keys())


class ModuleResolver:
    """Resolves, caches, and parses .axon module files.
    
    Uses lazy parsing — modules are only parsed on first access.
    Detects circular imports via a 'currently loading' set.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Args:
            base_dir: The directory to resolve relative imports against.
                      Defaults to the current working directory.
        """
        self.base_dir = base_dir or os.getcwd()
        self.registry = ModuleRegistry()
        self._loading: Set[str] = set()  # paths currently being loaded (cycle detection)

    def resolve_path(self, import_path: str, from_file: Optional[str] = None) -> str:
        """Resolve an import path to an absolute filesystem path.
        
        Args:
            import_path: The import path, e.g. './utils.axon' or '../lib/helpers.axon'
            from_file: The file that contains the import (for relative resolution)
        
        Returns:
            Absolute path to the .axon file.
        
        Raises:
            ModuleError: If the path cannot be resolved.
        """
        if from_file:
            base = os.path.dirname(os.path.abspath(from_file))
        else:
            base = self.base_dir

        # Resolve the path relative to the base
        if import_path.startswith(("./", "../", "/")):
            resolved = os.path.normpath(os.path.join(base, import_path))
        else:
            # Non-relative: resolve relative to base_dir
            resolved = os.path.normpath(os.path.join(self.base_dir, import_path))

        # Ensure .axon extension
        if not resolved.endswith(".axon"):
            resolved += ".axon"

        return resolved

    def load(self, import_path: str, from_file: Optional[str] = None) -> ModuleInfo:
        """Load and parse a module, with caching and cycle detection.
        
        Args:
            import_path: Import path string (relative or absolute)
            from_file: The importing file's path (for relative resolution)
        
        Returns:
            ModuleInfo with parsed definitions.
        
        Raises:
            CircularImportError: If a cycle is detected.
            ModuleError: If the file does not exist or cannot be parsed.
        """
        abs_path = self.resolve_path(import_path, from_file)

        # Return cached module if already loaded
        cached = self.registry.get_module(abs_path)
        if cached and cached.loaded:
            return cached

        # Detect circular imports
        if abs_path in self._loading:
            raise CircularImportError(
                f"Circular import detected: {abs_path} is already being loaded"
            )

        if not os.path.isfile(abs_path):
            raise ModuleError(f"Module not found: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Mark as loading
        self._loading.add(abs_path)

        try:
            definitions = self._parse_module(source, abs_path)
        finally:
            self._loading.discard(abs_path)

        info = ModuleInfo(
            path=abs_path,
            source=source,
            definitions=definitions,
            exports=list(definitions.keys()),
            loaded=True,
        )
        self.registry.register_module(info)
        return info

    def _parse_module(self, source: str, path: str) -> dict:
        """Parse an .axon source file and return a dict of {name: AST node}.
        
        Recursively processes imports within the module.
        """
        # Import here to avoid circular dependency at module load time
        from axon.parser.parser import AxonParser
        from axon.parser.ast_nodes import AxonImport

        parser = AxonParser(source)
        program = parser.parse()

        definitions = {}
        for node in program.definitions:
            # Recursively resolve sub-imports
            if isinstance(node, AxonImport) and node.source_path:
                try:
                    sub_info = self.load(node.source_path, from_file=path)
                    if node.import_style == "wildcard":
                        definitions.update(sub_info.definitions)
                    else:
                        for name in node.names:
                            if name in sub_info.definitions:
                                definitions[name] = sub_info.definitions[name]
                except ModuleError:
                    pass  # Missing sub-modules are silently ignored during indexing
            elif hasattr(node, "name") and node.name:
                definitions[node.name] = node

        return definitions

    def get_names(
        self,
        import_path: str,
        names: List[str],
        from_file: Optional[str] = None,
    ) -> dict:
        """Load a module and return specific named exports.
        
        Returns:
            {name: AST node} for each requested name found.
        """
        info = self.load(import_path, from_file)
        result = {}
        for name in names:
            node = info.definitions.get(name)
            if node is not None:
                result[name] = node
        return result

    def get_all(
        self, import_path: str, from_file: Optional[str] = None
    ) -> dict:
        """Load a module and return all its exports."""
        info = self.load(import_path, from_file)
        return dict(info.definitions)


def resolve_import(
    axon_import,  # AxonImport node
    resolver: ModuleResolver,
    from_file: Optional[str] = None,
) -> dict:
    """High-level helper: resolve an AxonImport node to a dict of {name: node}.
    
    Args:
        axon_import: An AxonImport AST node.
        resolver: A ModuleResolver instance.
        from_file: The file containing this import.
    
    Returns:
        Dict mapping imported names to their AST nodes.
    """
    from axon.parser.ast_nodes import AxonImport

    if not isinstance(axon_import, AxonImport):
        return {}
    if not axon_import.source_path:
        return {}

    if axon_import.import_style == "wildcard":
        return resolver.get_all(axon_import.source_path, from_file)
    elif axon_import.import_style == "named":
        return resolver.get_names(axon_import.source_path, axon_import.names, from_file)
    else:
        # python-style — not an axon import
        return {}
