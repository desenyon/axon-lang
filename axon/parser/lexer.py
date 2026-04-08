"""
Axon Lexer v2.0
=================
Tokenizer for the full Axon ML language.
Handles 40+ keywords, tensor annotations, indentation scoping,
and all expression types.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


class TokenType(Enum):
    # ─── Core ML Keywords ─────────────────────────────────────
    MODEL = auto()
    DATA = auto()
    TRAIN = auto()
    EVALUATE = auto()
    SEARCH = auto()
    DEPLOY = auto()
    PIPELINE = auto()
    TRANSFORM = auto()
    PRETRAIN = auto()
    FINETUNE = auto()
    ENSEMBLE = auto()
    EXPLAIN = auto()
    FORWARD = auto()
    
    # ─── Extended ML Keywords ─────────────────────────────────
    GAN = auto()
    DIFFUSION = auto()
    RL = auto()
    TABULAR = auto()
    TIMESERIES = auto()
    GRAPH = auto()
    AUDIO = auto()
    MULTIMODAL = auto()
    DISTILL = auto()
    QUANTIZE = auto()
    MONITOR = auto()
    SERVE = auto()
    TEST = auto()
    BENCHMARK = auto()
    AUGMENT = auto()
    FEATURE = auto()
    EMBEDDING = auto()
    TOKENIZER = auto()
    CALLBACK = auto()
    METRIC = auto()
    RAG = auto()
    AGENT = auto()
    FEDERATED = auto()
    AUTOML = auto()
    
    # ─── Structure Keywords ───────────────────────────────────
    PYTHON_BLOCK = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    
    # ─── Literals ─────────────────────────────────────────────
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    BOOL_TRUE = auto()
    BOOL_FALSE = auto()
    NONE = auto()
    
    # ─── Operators & Punctuation ──────────────────────────────
    COLON = auto()
    ARROW = auto()
    AT = auto()
    DOT = auto()
    COMMA = auto()
    SLASH = auto()
    EQUALS = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    PIPE = auto()
    STAR = auto()
    PLUS = auto()
    MINUS = auto()
    DOUBLE_DOT = auto()  # .. for ranges
    QUESTION = auto()    # ? for conditionals
    HASH = auto()        # # for comments
    PERCENT = auto()     # % for modulo
    AMPERSAND = auto()   # & for bitwise
    TILDE = auto()       # ~ for bitwise not
    
    # ─── Structure ────────────────────────────────────────────
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    
    # ─── Special ──────────────────────────────────────────────
    COMMENT = auto()
    DASH = auto()


KEYWORDS = {
    # Core
    "model": TokenType.MODEL,
    "data": TokenType.DATA,
    "train": TokenType.TRAIN,
    "evaluate": TokenType.EVALUATE,
    "search": TokenType.SEARCH,
    "deploy": TokenType.DEPLOY,
    "pipeline": TokenType.PIPELINE,
    "transform": TokenType.TRANSFORM,
    "pretrain": TokenType.PRETRAIN,
    "finetune": TokenType.FINETUNE,
    "ensemble": TokenType.ENSEMBLE,
    "explain": TokenType.EXPLAIN,
    "forward": TokenType.FORWARD,
    # Extended
    "gan": TokenType.GAN,
    "diffusion": TokenType.DIFFUSION,
    "rl": TokenType.RL,
    "tabular": TokenType.TABULAR,
    "timeseries": TokenType.TIMESERIES,
    "graph": TokenType.GRAPH,
    "audio": TokenType.AUDIO,
    "multimodal": TokenType.MULTIMODAL,
    "distill": TokenType.DISTILL,
    "quantize": TokenType.QUANTIZE,
    "monitor": TokenType.MONITOR,
    "serve": TokenType.SERVE,
    "test": TokenType.TEST,
    "benchmark": TokenType.BENCHMARK,
    "augment": TokenType.AUGMENT,
    "feature": TokenType.FEATURE,
    "embedding": TokenType.EMBEDDING,
    "tokenizer": TokenType.TOKENIZER,
    "callback": TokenType.CALLBACK,
    "metric": TokenType.METRIC,
    "rag": TokenType.RAG,
    "agent": TokenType.AGENT,
    "federated": TokenType.FEDERATED,
    "automl": TokenType.AUTOML,
    # Structure
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    # Booleans
    "true": TokenType.BOOL_TRUE,
    "True": TokenType.BOOL_TRUE,
    "false": TokenType.BOOL_FALSE,
    "False": TokenType.BOOL_FALSE,
    "none": TokenType.NONE,
    "None": TokenType.NONE,
    # Pass-through
    "self": TokenType.IDENTIFIER,
    "return": TokenType.IDENTIFIER,
    "if": TokenType.IDENTIFIER,
    "else": TokenType.IDENTIFIER,
    "for": TokenType.IDENTIFIER,
    "in": TokenType.IDENTIFIER,
    "with": TokenType.IDENTIFIER,
    "class": TokenType.IDENTIFIER,
    "def": TokenType.IDENTIFIER,
    "lambda": TokenType.IDENTIFIER,
    "yield": TokenType.IDENTIFIER,
    "async": TokenType.IDENTIFIER,
    "await": TokenType.IDENTIFIER,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


class AxonLexer:
    """
    Tokenizer for the Axon language v2.0.
    
    Handles 40+ ML keywords, indentation-based scoping,
    tensor type annotations, parameter references, arrow expressions,
    ratio expressions, range expressions, and all Python-compatible literals.
    """
    
    def __init__(self, source: str):
        self.source = source
        self.lines = source.split("\n")
        self.tokens: list[Token] = []
        self._bracket_depth = 0  # Track [, (, { nesting
        self._tokenize()
    
    def _tokenize(self):
        indent_stack = [0]
        
        for line_num, line in enumerate(self.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if stripped.startswith("#"):
                    self.tokens.append(Token(TokenType.COMMENT, stripped, line_num, 0))
                continue
            
            if stripped.startswith("@python"):
                self.tokens.append(Token(TokenType.PYTHON_BLOCK, stripped, line_num, 0))
                continue
            
            indent = len(line) - len(line.lstrip())
            
            # Only emit INDENT/DEDENT when not inside brackets
            if self._bracket_depth == 0:
                if indent > indent_stack[-1]:
                    indent_stack.append(indent)
                    self.tokens.append(Token(TokenType.INDENT, "", line_num, 0))
                else:
                    while indent < indent_stack[-1]:
                        indent_stack.pop()
                        self.tokens.append(Token(TokenType.DEDENT, "", line_num, 0))
            
            self._tokenize_line(stripped, line_num)
            self.tokens.append(Token(TokenType.NEWLINE, "\\n", line_num, len(stripped)))
        
        while len(indent_stack) > 1:
            indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, "", len(self.lines), 0))
        
        self.tokens.append(Token(TokenType.EOF, "", len(self.lines), 0))
    
    def _tokenize_line(self, line: str, line_num: int):
        i = 0
        while i < len(line):
            if line[i] in (" ", "\t"):
                i += 1
                continue
            
            if line[i] == "#":
                break
            
            # String literals (single, double, triple-quoted)
            if line[i] in ('"', "'"):
                # Check for triple quotes
                if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):
                    token, i = self._read_triple_string(line, i, line_num)
                else:
                    token, i = self._read_string(line, i, line_num)
                self.tokens.append(token)
                continue
            
            # Arrow ->
            if line[i:i+2] == "->":
                self.tokens.append(Token(TokenType.ARROW, "->", line_num, i))
                i += 2
                continue
            
            # Double dot ..
            if line[i:i+2] == "..":
                self.tokens.append(Token(TokenType.DOUBLE_DOT, "..", line_num, i))
                i += 2
                continue
            
            # Numbers
            if line[i].isdigit() or (line[i] == '-' and i + 1 < len(line) and line[i+1].isdigit()):
                if line[i] == '-' and self.tokens and self.tokens[-1].type in (
                    TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.RPAREN, TokenType.RBRACKET
                ):
                    self.tokens.append(Token(TokenType.MINUS, "-", line_num, i))
                    i += 1
                    continue
                token, i = self._read_number(line, i, line_num)
                self.tokens.append(token)
                continue
            
            # @ param reference
            if line[i] == "@":
                i += 1
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] == "_"):
                    i += 1
                self.tokens.append(Token(TokenType.AT, "@" + line[start:i], line_num, start - 1))
                continue
            
            # Identifiers and keywords
            if line[i].isalpha() or line[i] == "_":
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] == "_"):
                    i += 1
                word = line[start:i]
                token_type = KEYWORDS.get(word, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, word, line_num, start))
                continue
            
            # Multi-char and single-char tokens
            char_map = {
                ":": TokenType.COLON,
                ".": TokenType.DOT,
                ",": TokenType.COMMA,
                "/": TokenType.SLASH,
                "=": TokenType.EQUALS,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                "|": TokenType.PIPE,
                "*": TokenType.STAR,
                "+": TokenType.PLUS,
                "-": TokenType.DASH,
                "?": TokenType.QUESTION,
                "%": TokenType.PERCENT,
                "&": TokenType.AMPERSAND,
                "~": TokenType.TILDE,
            }
            
            if line[i] in char_map:
                # Track bracket depth for INDENT/DEDENT suppression
                if line[i] in ('(', '[', '{'):
                    self._bracket_depth += 1
                elif line[i] in (')', ']', '}'):
                    self._bracket_depth = max(0, self._bracket_depth - 1)
                self.tokens.append(Token(char_map[line[i]], line[i], line_num, i))
                i += 1
                continue
            
            i += 1
    
    def _read_string(self, line: str, start: int, line_num: int) -> tuple:
        quote = line[start]
        i = start + 1
        value = []
        while i < len(line) and line[i] != quote:
            if line[i] == "\\" and i + 1 < len(line):
                value.append(line[i:i+2])
                i += 2
            else:
                value.append(line[i])
                i += 1
        if i < len(line):
            i += 1
        return Token(TokenType.STRING, "".join(value), line_num, start), i
    
    def _read_triple_string(self, line: str, start: int, line_num: int) -> tuple:
        quote = line[start:start+3]
        i = start + 3
        value = []
        while i < len(line):
            if line[i:i+3] == quote:
                i += 3
                return Token(TokenType.STRING, "".join(value), line_num, start), i
            value.append(line[i])
            i += 1
        return Token(TokenType.STRING, "".join(value), line_num, start), i
    
    def _read_number(self, line: str, start: int, line_num: int) -> tuple:
        i = start
        if line[i] == '-':
            i += 1
        # Hex
        if i + 1 < len(line) and line[i] == '0' and line[i+1] in ('x', 'X'):
            i += 2
            while i < len(line) and line[i] in '0123456789abcdefABCDEF':
                i += 1
            return Token(TokenType.NUMBER, line[start:i], line_num, start), i
        # Decimal
        while i < len(line) and line[i].isdigit():
            i += 1
        if i < len(line) and line[i] == '.':
            i += 1
            while i < len(line) and line[i].isdigit():
                i += 1
        if i < len(line) and line[i] in ('e', 'E'):
            i += 1
            if i < len(line) and line[i] in ('+', '-'):
                i += 1
            while i < len(line) and line[i].isdigit():
                i += 1
        return Token(TokenType.NUMBER, line[start:i], line_num, start), i
    
    def get_tokens(self) -> list:
        return [t for t in self.tokens if t.type != TokenType.COMMENT]
    
    def __iter__(self) -> Iterator:
        return iter(self.get_tokens())
