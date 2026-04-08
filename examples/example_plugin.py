"""
Example Axon Plugin — "visualization" block type
=================================================

This file demonstrates how to build an Axon plugin that adds a new block
keyword (``visualization``) to the language.

Usage
-----
1.  Register the plugin programmatically::

        from axon.plugins import PluginRegistry
        from examples.example_plugin import VisualizationPlugin

        registry = PluginRegistry()
        plugin = VisualizationPlugin()
        plugin.register(registry)

2.  Or install the package and declare the entry_point::

        [options.entry_points]
        axon.plugins =
            visualization = examples.example_plugin:VisualizationPlugin

Then you can write .axon files that use the new block::

    visualization MyPlot:
        type: scatter
        x: "epoch"
        y: "loss"
        title: "Training Loss"
        backend: matplotlib

The plugin will transpile this into valid Python (matplotlib / plotly) code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from axon.plugins import AxonPlugin, LintRule, PluginRegistry


# ---------------------------------------------------------------------------
# AST node for visualization blocks
# ---------------------------------------------------------------------------


@dataclass
class VisualizationNode:
    """AST node produced when parsing a ``visualization`` block."""

    name: str
    body: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# The plugin itself
# ---------------------------------------------------------------------------


class VisualizationPlugin(AxonPlugin):
    """Adds the ``visualization`` block type to Axon.

    A ``visualization`` block describes a plot or dashboard panel.  It is
    transpiled to Python code that uses ``matplotlib`` by default or
    ``plotly`` when ``backend: plotly`` is specified.

    Supported properties
    --------------------
    type       : chart type — scatter | line | bar | histogram | heatmap
    x          : x-axis column/variable name (string)
    y          : y-axis column/variable name (string)
    title      : plot title (string)
    xlabel     : x-axis label (string, optional)
    ylabel     : y-axis label (string, optional)
    backend    : "matplotlib" (default) or "plotly"
    color      : line/marker colour (string, optional)
    save_path  : file path to save the figure (string, optional)
    """

    name = "axon-visualization"
    version = "1.0.0"
    block_types = ["visualization"]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_block(self, name: str, parser: Any) -> Optional[VisualizationNode]:
        """Parse a ``visualization BlockName: ...`` block."""
        # consume the 'visualization' IDENTIFIER token
        parser.advance()

        # next token should be the block name
        from axon.parser.lexer import TokenType

        name_token = parser.expect(TokenType.IDENTIFIER)
        block_name = name_token.value

        # expect colon + indented body
        parser.expect(TokenType.COLON)
        parser.skip_newlines()
        parser.expect(TokenType.INDENT)

        body: Dict[str, Any] = {}
        while not parser.at_block_end():
            parser.skip_newlines()
            if parser.at_block_end():
                break
            key_token = parser.peek()
            if key_token.type == TokenType.IDENTIFIER:
                parser.advance()
                parser.expect(TokenType.COLON)
                value = parser.advance()
                body[key_token.value] = value.value
            else:
                parser.advance()  # skip unexpected token

        parser.match(TokenType.DEDENT)

        return VisualizationNode(
            name=block_name,
            body=body,
            line=name_token.line,
            col=name_token.col,
        )

    # ------------------------------------------------------------------
    # Transpilation
    # ------------------------------------------------------------------

    def transpile_block(self, node: VisualizationNode, transpiler: Any) -> str:
        """Transpile a :class:`VisualizationNode` to Python."""
        if not isinstance(node, VisualizationNode):
            return ""

        body = node.body
        chart_type = body.get("type", "line")
        x = body.get("x", "x")
        y = body.get("y", "y")
        title = body.get("title", node.name)
        xlabel = body.get("xlabel", x)
        ylabel = body.get("ylabel", y)
        color = body.get("color", "blue")
        backend = body.get("backend", "matplotlib")
        save_path = body.get("save_path", "")

        fn_name = f"visualize_{node.name.lower()}"

        if backend == "plotly":
            return self._transpile_plotly(
                fn_name, chart_type, x, y, title, xlabel, ylabel, save_path
            )
        return self._transpile_matplotlib(
            fn_name, chart_type, x, y, title, xlabel, ylabel, color, save_path
        )

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def get_completions(self, context: Any) -> List[str]:
        return [
            "visualization",
            "type",
            "scatter",
            "line",
            "bar",
            "histogram",
            "heatmap",
            "backend",
            "matplotlib",
            "plotly",
            "title",
            "xlabel",
            "ylabel",
            "save_path",
        ]

    # ------------------------------------------------------------------
    # Lint rules
    # ------------------------------------------------------------------

    def get_lint_rules(self) -> List[LintRule]:
        def check_type(node: Any) -> List[str]:
            if isinstance(node, VisualizationNode):
                allowed = {"scatter", "line", "bar", "histogram", "heatmap"}
                chart = node.body.get("type", "line")
                if chart not in allowed:
                    return [
                        f"Unknown visualization type '{chart}'. "
                        f"Allowed: {', '.join(sorted(allowed))}"
                    ]
            return []

        return [
            LintRule(
                rule_id="VIZ001",
                description="visualization block must have a valid 'type' property",
                severity="error",
                check=check_type,
            ),
            LintRule(
                rule_id="VIZ002",
                description="visualization block should specify 'x' and 'y' properties",
                severity="warning",
                check=lambda node: (
                    ["Missing 'x' or 'y' in visualization block"]
                    if isinstance(node, VisualizationNode)
                    and ("x" not in node.body or "y" not in node.body)
                    else []
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transpile_matplotlib(
        fn_name: str,
        chart_type: str,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str,
        save_path: str,
    ) -> str:
        plot_call = {
            "scatter": f"plt.scatter({x}, {y}, color='{color}')",
            "bar": f"plt.bar({x}, {y}, color='{color}')",
            "histogram": f"plt.hist({y}, color='{color}')",
            "heatmap": f"sns.heatmap({y})",
        }.get(chart_type, f"plt.plot({x}, {y}, color='{color}')")

        save_code = f"    plt.savefig('{save_path}')\n" if save_path else ""

        return (
            f"import matplotlib.pyplot as plt\n"
            f"try:\n"
            f"    import seaborn as sns\n"
            f"except ImportError:\n"
            f"    pass\n\n"
            f"def {fn_name}({x}, {y}):\n"
            f"    \"\"\"Auto-generated visualization: {title}\"\"\"\n"
            f"    {plot_call}\n"
            f"    plt.title('{title}')\n"
            f"    plt.xlabel('{xlabel}')\n"
            f"    plt.ylabel('{ylabel}')\n"
            f"{save_code}"
            f"    plt.show()\n"
        )

    @staticmethod
    def _transpile_plotly(
        fn_name: str,
        chart_type: str,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: str,
    ) -> str:
        trace_call = {
            "scatter": f"go.Scatter(x={x}, y={y}, mode='markers')",
            "bar": f"go.Bar(x={x}, y={y})",
            "histogram": f"go.Histogram(x={y})",
            "heatmap": f"go.Heatmap(z={y})",
        }.get(chart_type, f"go.Scatter(x={x}, y={y})")

        save_code = f"    fig.write_html('{save_path}')\n" if save_path else ""

        return (
            f"import plotly.graph_objects as go\n\n"
            f"def {fn_name}({x}, {y}):\n"
            f"    \"\"\"Auto-generated visualization: {title}\"\"\"\n"
            f"    fig = go.Figure(data=[{trace_call}])\n"
            f"    fig.update_layout(title='{title}', "
            f"xaxis_title='{xlabel}', yaxis_title='{ylabel}')\n"
            f"{save_code}"
            f"    fig.show()\n"
        )


# ---------------------------------------------------------------------------
# Convenience: plugin manifest example
# ---------------------------------------------------------------------------

EXAMPLE_MANIFEST = {
    "name": "axon-visualization",
    "version": "1.0.0",
    "block_types": ["visualization"],
    "entry_point": "examples.example_plugin:VisualizationPlugin",
}

# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    registry = PluginRegistry()
    plugin = VisualizationPlugin()
    plugin.register(registry)

    print("Loaded plugins:")
    for info in registry.list_plugins():
        print(f"  {info['name']} v{info['version']}  blocks={info['block_types']}")

    print("\nExample manifest:")
    print(json.dumps(EXAMPLE_MANIFEST, indent=2))

    print("\nPlugin completions:")
    print(registry.get_all_completions())
