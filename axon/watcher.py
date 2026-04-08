"""
Axon Watch Mode
================
Monitors .axon files for changes and recompiles on modification.

Uses polling (os.stat mtime checks) — no external dependencies required.
Supports watching a single file or a directory tree recursively.

Usage:
    watcher = AxonWatcher("model.axon", backend="pytorch")
    watcher.start()   # blocking
    
    watcher = AxonWatcher("./src", recursive=True, backend="jax")
    watcher.start()
"""

import os
import sys
import time
from typing import Optional, Dict, List, Callable

# ─── ANSI color codes ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"
_DIM    = "\033[2m"


def _color(text: str, *codes: str) -> str:
    """Wrap text in ANSI color codes."""
    return "".join(codes) + text + _RESET


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _print_status(msg: str, status: str = "info"):
    """Print a colored status message."""
    ts = _color(f"[{_timestamp()}]", _DIM)
    if status == "ok":
        icon = _color("✓", _GREEN, _BOLD)
        line = f"{ts} {icon} {msg}"
    elif status == "error":
        icon = _color("✗", _RED, _BOLD)
        line = f"{ts} {icon} {_color(msg, _RED)}"
    elif status == "change":
        icon = _color("~", _CYAN, _BOLD)
        line = f"{ts} {icon} {_color(msg, _CYAN)}"
    elif status == "watch":
        icon = _color("◉", _BLUE, _BOLD)
        line = f"{ts} {icon} {_color(msg, _BLUE)}"
    else:
        icon = _color("·", _YELLOW)
        line = f"{ts} {icon} {msg}"
    print(line, flush=True)


class AxonWatcher:
    """
    Watches .axon files for modifications and triggers recompilation.
    
    Attributes:
        path: File or directory path to watch.
        backend: Compilation backend ('pytorch', 'tensorflow', 'jax').
        output_dir: Output directory for compiled files.
        recursive: If path is a directory, watch subdirectories too.
        poll_interval: Seconds between mtime checks (default 0.5).
        on_compile: Optional callback(filepath, success, output) called after each compile.
    """

    def __init__(
        self,
        path: str,
        backend: str = "pytorch",
        output_dir: str = "./axon_output",
        recursive: bool = True,
        poll_interval: float = 0.5,
        on_compile: Optional[Callable[[str, bool, str], None]] = None,
    ):
        self.path = os.path.abspath(path)
        self.backend = backend
        self.output_dir = output_dir
        self.recursive = recursive
        self.poll_interval = poll_interval
        self.on_compile = on_compile

        # mtime cache: absolute_path -> last mtime
        self._mtimes: Dict[str, float] = {}
        self._running = False

    # ─── File Discovery ──────────────────────────────────────────────────────

    def _find_axon_files(self) -> List[str]:
        """Return a list of absolute paths to all .axon files being watched."""
        if os.path.isfile(self.path):
            if self.path.endswith(".axon"):
                return [self.path]
            return []
        elif os.path.isdir(self.path):
            result = []
            if self.recursive:
                for root, dirs, files in os.walk(self.path):
                    # Skip hidden directories and __pycache__
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                    for fname in files:
                        if fname.endswith(".axon"):
                            result.append(os.path.join(root, fname))
            else:
                for fname in os.listdir(self.path):
                    if fname.endswith(".axon"):
                        result.append(os.path.join(self.path, fname))
            return result
        return []

    def _get_mtime(self, filepath: str) -> float:
        """Return the mtime of a file, or 0.0 if it doesn't exist."""
        try:
            return os.stat(filepath).st_mtime
        except OSError:
            return 0.0

    def _init_mtimes(self):
        """Initialize mtime cache for all currently watched files."""
        for fpath in self._find_axon_files():
            self._mtimes[fpath] = self._get_mtime(fpath)

    # ─── Compilation ─────────────────────────────────────────────────────────

    def _compile_file(self, filepath: str) -> tuple:
        """Compile a single .axon file.
        
        Returns:
            (success: bool, output_path: str, error: str)
        """
        try:
            from axon.runtime.executor import AxonExecutor
            from axon.runtime.config import AxonConfig

            config = AxonConfig(
                backend=self.backend,
                output_dir=self.output_dir,
            )
            executor = AxonExecutor(config=config)
            python_code = executor.compile_file(filepath)

            # Write output
            os.makedirs(self.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(self.output_dir, f"{base}.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(python_code)

            return True, out_path, ""
        except Exception as e:
            return False, "", str(e)

    def _handle_change(self, filepath: str):
        """Called when a file change is detected."""
        rel = os.path.relpath(filepath)
        _print_status(f"Changed: {rel}", "change")

        success, out_path, error = self._compile_file(filepath)

        if success:
            rel_out = os.path.relpath(out_path)
            _print_status(f"Compiled → {rel_out}", "ok")
        else:
            _print_status(f"Compile error: {error}", "error")

        if self.on_compile:
            self.on_compile(filepath, success, out_path if success else error)

    # ─── Poll Loop ───────────────────────────────────────────────────────────

    def _check_changes(self) -> List[str]:
        """Check for changed/new files. Returns list of changed paths."""
        changed = []
        current_files = set(self._find_axon_files())

        # Check existing files for modifications
        for fpath in current_files:
            mtime = self._get_mtime(fpath)
            if fpath not in self._mtimes:
                # New file detected
                self._mtimes[fpath] = mtime
                changed.append(fpath)
            elif mtime != self._mtimes[fpath]:
                self._mtimes[fpath] = mtime
                changed.append(fpath)

        # Clean up deleted files from cache
        deleted = set(self._mtimes.keys()) - current_files
        for fpath in deleted:
            del self._mtimes[fpath]

        return changed

    def start(self):
        """Start watching — blocks until KeyboardInterrupt."""
        self._running = True
        self._init_mtimes()

        watched_files = self._find_axon_files()
        if not watched_files:
            _print_status(f"No .axon files found at {self.path}", "error")
            return

        n = len(watched_files)
        desc = "recursively" if self.recursive and os.path.isdir(self.path) else ""
        _print_status(
            f"Watching {n} file{'s' if n != 1 else ''} {desc} in "
            f"{_color(os.path.relpath(self.path), _BOLD)} "
            f"[backend={_color(self.backend, _CYAN)}]",
            "watch"
        )
        _print_status("Press Ctrl+C to stop.", "info")

        # Initial compile of all files
        for fpath in watched_files:
            success, out_path, error = self._compile_file(fpath)
            rel = os.path.relpath(fpath)
            if success:
                rel_out = os.path.relpath(out_path)
                _print_status(f"Initial compile: {rel} → {rel_out}", "ok")
            else:
                _print_status(f"Initial compile failed: {rel}: {error}", "error")

        try:
            while self._running:
                time.sleep(self.poll_interval)
                changed = self._check_changes()
                for fpath in changed:
                    self._handle_change(fpath)
        except KeyboardInterrupt:
            _print_status("Watch mode stopped.", "info")
        finally:
            self._running = False

    def stop(self):
        """Stop the watcher (can be called from another thread)."""
        self._running = False

    def poll_once(self) -> List[str]:
        """Check for changes once and return list of changed file paths.
        
        Useful for testing — does not sleep or loop.
        """
        return self._check_changes()
