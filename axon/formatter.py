"""
Axon Code Formatter
====================
Formats Axon source code with consistent style rules:
- 4-space indentation
- One blank line between top-level blocks
- No trailing whitespace
- Sorted properties (priority fields first, then alphabetical)
- Colon alignment within blocks
- Double-quote normalization for strings
- Max 2 consecutive blank lines
- Trailing newline
"""

import re
import difflib
from typing import List, Tuple, Optional

from axon.parser.lexer import AxonLexer, TokenType, Token


# Priority ordering for common property keys (appear first, then rest alphabetical)
PRIORITY_KEYS = ["name", "model", "data", "source", "format", "split",
                 "optimizer", "loss", "epochs", "lr", "batch_size",
                 "layers", "transform", "loader", "metrics", "device"]


class AxonFormatter:
    """
    Format Axon source code into a canonical style.

    Works at the line/token level for robustness — does NOT require
    a fully-valid parse tree.
    """

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()

    # ─── Public API ───────────────────────────────────────────────

    def format(self) -> str:
        """Return formatted Axon source code."""
        lines = self._normalize_strings(self.lines)
        lines = self._normalize_indentation(lines)
        lines = self._strip_trailing_whitespace(lines)
        lines = self._sort_block_properties(lines)
        lines = self._align_colons(lines)
        lines = self._normalize_blank_lines(lines)
        result = "\n".join(lines)
        if not result.endswith("\n"):
            result += "\n"
        return result

    # ─── String Normalization ────────────────────────────────────

    def _normalize_strings(self, lines: List[str]) -> List[str]:
        """Convert single-quoted strings to double-quoted strings."""
        out = []
        for line in lines:
            out.append(self._convert_quotes(line))
        return out

    def _convert_quotes(self, line: str) -> str:
        """Replace 'string' with "string" in a line, safely."""
        result = []
        i = 0
        in_double = False
        while i < len(line):
            ch = line[i]
            # Skip double-quoted strings
            if ch == '"' and not in_double:
                in_double = True
                result.append(ch)
                i += 1
                continue
            if ch == '"' and in_double:
                in_double = False
                result.append(ch)
                i += 1
                continue
            if in_double:
                if ch == '\\':
                    result.append(ch)
                    if i + 1 < len(line):
                        result.append(line[i + 1])
                        i += 2
                    else:
                        i += 1
                    continue
                result.append(ch)
                i += 1
                continue
            # Single-quoted string
            if ch == "'":
                # Find end of single-quoted string
                j = i + 1
                content = []
                while j < len(line) and line[j] != "'":
                    if line[j] == '\\' and j + 1 < len(line):
                        content.append(line[j])
                        content.append(line[j + 1])
                        j += 2
                    else:
                        content.append(line[j])
                        j += 1
                inner = "".join(content)
                # Escape any double quotes inside
                inner = inner.replace('"', '\\"')
                result.append('"')
                result.append(inner)
                result.append('"')
                i = j + 1  # skip closing quote
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    # ─── Indentation Normalization ───────────────────────────────

    def _normalize_indentation(self, lines: List[str]) -> List[str]:
        """Replace any tab-based or inconsistent indentation with 4 spaces."""
        out = []
        for line in lines:
            if line.strip() == "":
                out.append("")
                continue
            stripped = line.lstrip()
            raw_indent = line[: len(line) - len(stripped)]
            # Count indentation level
            # Convert tabs to spaces (1 tab = 4 spaces)
            normalized = raw_indent.replace("\t", "    ")
            # Round to nearest 4-space multiple
            level = len(normalized) // 4
            out.append("    " * level + stripped)
        return out

    # ─── Trailing Whitespace ─────────────────────────────────────

    def _strip_trailing_whitespace(self, lines: List[str]) -> List[str]:
        return [line.rstrip() for line in lines]

    # ─── Property Sorting within Blocks ──────────────────────────

    def _sort_block_properties(self, lines: List[str]) -> List[str]:
        """
        For each indented block, sort the key: value lines.
        Priority keys come first (in PRIORITY_KEYS order),
        then remaining keys sorted alphabetically.
        Sub-blocks (lines followed by indented content) are preserved as-is.
        """
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Detect a block header: top-level keyword + name + colon
            if self._is_block_header(stripped) and not line.startswith("    "):
                out.append(line)
                i += 1
                # Collect the block body (indented lines)
                block_lines = []
                while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                    block_lines.append(lines[i])
                    i += 1
                sorted_block = self._sort_block(block_lines)
                out.extend(sorted_block)
            else:
                out.append(line)
                i += 1
        return out

    def _is_block_header(self, stripped: str) -> bool:
        """Check if a stripped line looks like an Axon block header."""
        block_keywords = {
            "model", "data", "train", "evaluate", "search", "deploy",
            "pipeline", "transform", "pretrain", "finetune", "ensemble",
            "explain", "forward", "gan", "diffusion", "rl", "tabular",
            "timeseries", "graph", "audio", "multimodal", "distill",
            "quantize", "monitor", "serve", "test", "benchmark", "augment",
            "feature", "embedding", "tokenizer", "callback", "metric",
            "rag", "agent", "federated", "automl",
        }
        parts = stripped.split()
        return (len(parts) >= 2 and parts[0] in block_keywords
                and stripped.endswith(":"))

    def _sort_block(self, block_lines: List[str]) -> List[str]:
        """
        Sort simple key: value lines within a block.
        Sub-blocks (lines whose next line is more indented) are kept together.
        Blank lines between groups are preserved.
        """
        if not block_lines:
            return block_lines

        # Group into segments: simple key-value pairs vs sub-blocks
        segments = self._group_segments(block_lines)
        # Sort simple KV segments
        kv_segments = []
        other_segments = []

        for seg in segments:
            if seg["type"] == "kv":
                kv_segments.append(seg)
            else:
                other_segments.append(seg)

        # Sort kv segments by priority then alpha
        def sort_key(seg):
            k = seg["key"].lower()
            try:
                return (0, PRIORITY_KEYS.index(k), k)
            except ValueError:
                return (1, 0, k)

        kv_segments.sort(key=sort_key)

        # Reconstruct: kv segments first, then sub-blocks
        result = []
        for seg in kv_segments:
            result.extend(seg["lines"])
        for seg in other_segments:
            result.extend(seg["lines"])

        return result

    def _group_segments(self, lines: List[str]) -> List[dict]:
        """Group block lines into simple kv pairs and sub-blocks."""
        segments = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped == "":
                i += 1
                continue
            # Check if next line is more indented (sub-block)
            curr_indent = len(line) - len(line.lstrip()) if line.strip() else 0
            next_indent = 0
            if i + 1 < len(lines) and lines[i + 1].strip():
                next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())

            if next_indent > curr_indent:
                # This is a sub-block header + its indented body
                sub_lines = [line]
                i += 1
                while i < len(lines):
                    sub_indent = len(lines[i]) - len(lines[i].lstrip()) if lines[i].strip() else 0
                    if lines[i].strip() == "" or sub_indent > curr_indent:
                        sub_lines.append(lines[i])
                        i += 1
                    else:
                        break
                segments.append({"type": "sub", "lines": sub_lines, "key": ""})
            else:
                # Simple line
                key = stripped.split(":")[0].strip() if ":" in stripped else stripped
                segments.append({"type": "kv", "lines": [line], "key": key})
                i += 1
        return segments

    # ─── Colon Alignment ─────────────────────────────────────────

    def _align_colons(self, lines: List[str]) -> List[str]:
        """
        Within each block, align colons of simple key: value lines.
        Finds the longest key and pads all others to match.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            if self._is_block_header(out[i].strip()) and not out[i].startswith("    "):
                i += 1
                # Find the span of the direct children (one level of indentation = 4 spaces)
                block_start = i
                while i < len(out) and (out[i].startswith("    ") or out[i].strip() == ""):
                    i += 1
                block_end = i
                self._align_block_colons(out, block_start, block_end)
            else:
                i += 1
        return out

    def _align_block_colons(self, lines: List[str], start: int, end: int):
        """Align colon positions for direct children in lines[start:end]."""
        # Only look at lines with exactly 4-space indent (direct children)
        child_indices = []
        max_key_len = 0
        for idx in range(start, end):
            line = lines[idx]
            if not line.startswith("    ") or line.startswith("        "):
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Must have a colon and look like key: value (not a sub-block header)
            m = re.match(r'^(\w[\w\s]*?)\s*:\s*(.*)$', stripped)
            if m and not self._is_block_header(stripped):
                key_len = len(m.group(1))
                if key_len > max_key_len:
                    max_key_len = key_len
                child_indices.append((idx, m.group(1), m.group(2)))

        if not child_indices or max_key_len == 0:
            return

        for idx, key, value in child_indices:
            padding = " " * (max_key_len - len(key))
            if value:
                lines[idx] = f"    {key}{padding}: {value}"
            else:
                lines[idx] = f"    {key}{padding}:"

    # ─── Blank Line Normalization ─────────────────────────────────

    def _normalize_blank_lines(self, lines: List[str]) -> List[str]:
        """
        - Max 2 consecutive blank lines anywhere
        - Exactly 1 blank line between top-level blocks
        - No leading/trailing blank lines
        """
        # First pass: max 2 consecutive blanks
        out = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    out.append("")
            else:
                blank_count = 0
                out.append(line)

        # Second pass: ensure exactly 1 blank line between top-level blocks
        result = []
        i = 0
        while i < len(out):
            line = out[i]
            if (self._is_block_header(line.strip()) and
                    not line.startswith(" ") and not line.startswith("\t")):
                # Remove trailing blanks before this block header (we'll add exactly 1)
                while result and result[-1].strip() == "":
                    result.pop()
                if result:  # not the first block
                    result.append("")
                result.append(line)
            else:
                result.append(line)
            i += 1

        # Remove leading and trailing blank lines
        while result and result[0].strip() == "":
            result.pop(0)
        while result and result[-1].strip() == "":
            result.pop()

        return result


def format_source(source: str) -> str:
    """Convenience function to format Axon source code."""
    return AxonFormatter(source).format()


def format_diff(original: str, formatted: str, filename: str = "<axon>") -> str:
    """Return a unified diff between original and formatted source."""
    original_lines = original.splitlines(keepends=True)
    formatted_lines = formatted.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        formatted_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)
