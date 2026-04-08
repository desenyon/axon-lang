"""
Axon Parser v2.0
==================
Recursive descent parser for all 30+ Axon block types.
Converts token streams into a comprehensive AST covering
the entire ML/AI ecosystem.
"""

from typing import Any, Optional
from axon.parser.lexer import AxonLexer, Token, TokenType
from axon.parser.ast_nodes import *


class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at line {token.line}:{token.col}" if token else ""
        super().__init__(f"ParseError{loc}: {message}")


class AxonParser:
    """Parser for Axon v2.0 — handles 30+ block types."""
    
    def __init__(self, source: str):
        self.lexer = AxonLexer(source)
        self.tokens = self.lexer.get_tokens()
        self.pos = 0
        self.source = source
    
    # ─── Token Navigation ──────────────────────────────────────
    
    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token(TokenType.EOF, "", 0, 0)
    
    def advance(self) -> Token:
        token = self.peek()
        self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.peek()
        if token.type != token_type:
            raise ParseError(f"Expected {token_type.name}, got {token.type.name} ({token.value!r})", token)
        return self.advance()
    
    def match(self, *types: TokenType) -> Optional[Token]:
        if self.peek().type in types:
            return self.advance()
        return None
    
    def skip_newlines(self):
        while self.peek().type == TokenType.NEWLINE:
            self.advance()
    
    def at_block_end(self) -> bool:
        t = self.peek()
        return t.type in (TokenType.DEDENT, TokenType.EOF)
    
    # ─── Top-Level ─────────────────────────────────────────────
    
    def parse(self) -> Program:
        program = Program()
        self.skip_newlines()
        while self.peek().type != TokenType.EOF:
            self.skip_newlines()
            if self.peek().type == TokenType.EOF:
                break
            node = self._parse_top_level()
            if node:
                program.definitions.append(node)
            self.skip_newlines()
        return program
    
    def _parse_top_level(self) -> Optional[ASTNode]:
        token = self.peek()
        
        # Map all block-starting keywords to their parsers
        dispatch = {
            TokenType.MODEL: self._parse_model,
            TokenType.DATA: self._parse_data,
            TokenType.TRAIN: self._parse_train,
            TokenType.EVALUATE: self._parse_evaluate,
            TokenType.SEARCH: self._parse_search,
            TokenType.DEPLOY: self._parse_deploy,
            TokenType.PIPELINE: self._parse_pipeline,
            TokenType.FINETUNE: self._parse_finetune,
            TokenType.ENSEMBLE: self._parse_ensemble,
            TokenType.EXPLAIN: self._parse_explain,
            TokenType.PRETRAIN: self._parse_pretrain,
            TokenType.GAN: self._parse_gan,
            TokenType.DIFFUSION: self._parse_diffusion,
            TokenType.RL: self._parse_rl,
            TokenType.TABULAR: self._parse_tabular,
            TokenType.TIMESERIES: self._parse_timeseries,
            TokenType.GRAPH: self._parse_graph,
            TokenType.AUDIO: self._parse_audio,
            TokenType.MULTIMODAL: self._parse_multimodal,
            TokenType.DISTILL: self._parse_distill,
            TokenType.QUANTIZE: self._parse_quantize,
            TokenType.MONITOR: self._parse_monitor,
            TokenType.SERVE: self._parse_serve,
            TokenType.TEST: self._parse_test,
            TokenType.BENCHMARK: self._parse_benchmark,
            TokenType.AUGMENT: self._parse_augment,
            TokenType.FEATURE: self._parse_feature,
            TokenType.EMBEDDING: self._parse_embedding,
            TokenType.TOKENIZER: self._parse_tokenizer,
            TokenType.CALLBACK: self._parse_callback,
            TokenType.METRIC: self._parse_metric,
            TokenType.RAG: self._parse_rag,
            TokenType.AGENT: self._parse_agent,
            TokenType.FEDERATED: self._parse_federated,
            TokenType.AUTOML: self._parse_automl,
            TokenType.PYTHON_BLOCK: self._parse_python_block,
            TokenType.IMPORT: self._parse_import,
            TokenType.FROM: self._parse_import,
        }
        
        parser_fn = dispatch.get(token.type)
        if parser_fn:
            return parser_fn()
        
        # Skip unknown tokens silently only for NEWLINE/INDENT/DEDENT/EOF
        if token.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
            self.advance()
            return None
        
        raise ParseError(f"Unexpected token '{token.value}' ({token.type.name}) — expected a block keyword (model, data, train, etc.)", token)
    
    # ─── Value Parsing ─────────────────────────────────────────
    
    def _parse_value(self) -> Any:
        token = self.peek()
        
        if token.type == TokenType.STRING:
            self.advance()
            return StringLiteral(value=token.value, line=token.line, col=token.col)
        
        if token.type == TokenType.NUMBER:
            self.advance()
            val = float(token.value) if ('.' in token.value or 'e' in token.value.lower()) and not token.value.startswith('0x') else int(token.value, 0)
            if self.peek().type == TokenType.SLASH:
                parts = [val]
                while self.match(TokenType.SLASH):
                    next_num = self.expect(TokenType.NUMBER)
                    nval = float(next_num.value) if '.' in next_num.value else int(next_num.value)
                    parts.append(nval)
                return RatioExpr(parts=parts, line=token.line, col=token.col)
            return NumberLiteral(value=val, line=token.line, col=token.col)
        
        if token.type == TokenType.BOOL_TRUE:
            self.advance()
            return BoolLiteral(value=True, line=token.line, col=token.col)
        if token.type == TokenType.BOOL_FALSE:
            self.advance()
            return BoolLiteral(value=False, line=token.line, col=token.col)
        
        if token.type == TokenType.NONE:
            self.advance()
            return Identifier(name="None", line=token.line, col=token.col)
        
        if token.type == TokenType.AT:
            self.advance()
            return ParamRef(name=token.value[1:], line=token.line, col=token.col)
        
        if token.type == TokenType.LBRACKET:
            return self._parse_list()
        
        if token.type == TokenType.LBRACE:
            return self._parse_dict()
        
        if token.type == TokenType.IDENTIFIER:
            return self._parse_identifier_or_call()
        
        # Keywords used as identifiers in value position
        keyword_as_id = {
            TokenType.MODEL, TokenType.DATA, TokenType.TRAIN, TokenType.EVALUATE,
            TokenType.TRANSFORM, TokenType.FORWARD, TokenType.DEPLOY, TokenType.SERVE,
            TokenType.TEST, TokenType.FEATURE, TokenType.GRAPH, TokenType.AUDIO,
            TokenType.AGENT, TokenType.METRIC, TokenType.EMBEDDING, TokenType.MONITOR,
            TokenType.GAN, TokenType.DIFFUSION, TokenType.RL, TokenType.TABULAR,
            TokenType.TIMESERIES, TokenType.MULTIMODAL, TokenType.DISTILL,
            TokenType.QUANTIZE, TokenType.BENCHMARK, TokenType.AUGMENT,
            TokenType.TOKENIZER, TokenType.CALLBACK, TokenType.RAG,
            TokenType.FEDERATED, TokenType.AUTOML, TokenType.PIPELINE,
            TokenType.PRETRAIN, TokenType.FINETUNE, TokenType.ENSEMBLE,
            TokenType.EXPLAIN, TokenType.SEARCH,
        }
        if token.type in keyword_as_id:
            self.advance()
            node = Identifier(name=token.value, line=token.line, col=token.col)
            while self.peek().type == TokenType.DOT:
                self.advance()
                attr_token = self.advance()
                node = AttributeAccess(obj=node, attr=attr_token.value, line=attr_token.line, col=attr_token.col)
            if self.peek().type == TokenType.LPAREN:
                return self._parse_function_call(node)
            return node
        
        self.advance()
        return Identifier(name=token.value, line=token.line, col=token.col)
    
    def _parse_identifier_or_call(self) -> ASTNode:
        token = self.advance()
        node = Identifier(name=token.value, line=token.line, col=token.col)
        
        while self.peek().type == TokenType.DOT:
            self.advance()
            attr = self.advance()
            node = AttributeAccess(obj=node, attr=attr.value, line=attr.line, col=attr.col)
        
        if self.peek().type == TokenType.LPAREN:
            return self._parse_function_call(node)
        
        return node
    
    def _parse_function_call(self, name_node: ASTNode) -> FunctionCall:
        name = name_node.name if isinstance(name_node, Identifier) else str(name_node)
        if isinstance(name_node, AttributeAccess):
            parts = []
            n = name_node
            while isinstance(n, AttributeAccess):
                parts.append(n.attr)
                n = n.obj
            if isinstance(n, Identifier):
                parts.append(n.name)
            name = ".".join(reversed(parts))
        
        self.expect(TokenType.LPAREN)
        args = []
        kwargs = {}
        
        self._skip_whitespace_tokens()
        while self.peek().type != TokenType.RPAREN and self.peek().type != TokenType.EOF:
            self._skip_whitespace_tokens()
            if self.peek().type == TokenType.RPAREN:
                break
            if (self.peek().type == TokenType.IDENTIFIER and self.peek(1).type == TokenType.EQUALS):
                key = self.advance().value
                self.advance()
                val = self._parse_value()
                kwargs[key] = val
            else:
                val = self._parse_value()
                if self.peek().type == TokenType.ARROW:
                    self.advance()
                    right = self._parse_value()
                    val = ArrowExpr(left=val, right=right)
                args.append(val)
            self._skip_whitespace_tokens()
            self.match(TokenType.COMMA)
            self._skip_whitespace_tokens()
        
        self.expect(TokenType.RPAREN)
        return FunctionCall(name=name, args=args, kwargs=kwargs, line=name_node.line, col=name_node.col)
    
    def _skip_whitespace_tokens(self):
        """Skip NEWLINE, INDENT, DEDENT tokens (used inside brackets/parens)."""
        while self.peek().type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
            self.advance()
    
    def _parse_list(self) -> ListLiteral:
        token = self.expect(TokenType.LBRACKET)
        elements = []
        self._skip_whitespace_tokens()
        while self.peek().type != TokenType.RBRACKET and self.peek().type != TokenType.EOF:
            self._skip_whitespace_tokens()
            if self.peek().type == TokenType.RBRACKET:
                break
            elements.append(self._parse_value())
            self._skip_whitespace_tokens()
            self.match(TokenType.COMMA)
            self._skip_whitespace_tokens()
        self.expect(TokenType.RBRACKET)
        return ListLiteral(elements=elements, line=token.line, col=token.col)
    
    def _parse_dict(self) -> DictLiteral:
        token = self.expect(TokenType.LBRACE)
        pairs = []
        self._skip_whitespace_tokens()
        while self.peek().type != TokenType.RBRACE and self.peek().type != TokenType.EOF:
            self._skip_whitespace_tokens()
            if self.peek().type == TokenType.RBRACE:
                break
            key = self._parse_value()
            self.expect(TokenType.COLON)
            val = self._parse_value()
            pairs.append((key, val))
            self._skip_whitespace_tokens()
            self.match(TokenType.COMMA)
            self._skip_whitespace_tokens()
        self.expect(TokenType.RBRACE)
        return DictLiteral(pairs=pairs, line=token.line, col=token.col)
    
    # ─── Block Parsing Helpers ─────────────────────────────────
    
    def _parse_block_header(self, keyword_type: TokenType) -> str:
        self.expect(keyword_type)
        name_token = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COLON)
        self.skip_newlines()
        return name_token.value
    
    def _parse_key_value_block(self) -> dict:
        result = {}
        if self.peek().type != TokenType.INDENT:
            return result
        self.advance()
        
        while not self.at_block_end():
            self.skip_newlines()
            if self.at_block_end():
                break
            
            key_token = self.peek()
            
            # Handle all keyword types that can appear as keys
            keyword_key_types = {
                TokenType.IDENTIFIER, TokenType.MODEL, TokenType.DATA, TokenType.TRAIN,
                TokenType.EVALUATE, TokenType.TRANSFORM, TokenType.FORWARD, TokenType.DEPLOY,
                TokenType.SERVE, TokenType.TEST, TokenType.FEATURE, TokenType.GRAPH,
                TokenType.AUDIO, TokenType.AGENT, TokenType.METRIC, TokenType.EMBEDDING,
                TokenType.MONITOR, TokenType.GAN, TokenType.DIFFUSION, TokenType.RL,
                TokenType.TABULAR, TokenType.TIMESERIES, TokenType.MULTIMODAL,
                TokenType.DISTILL, TokenType.QUANTIZE, TokenType.BENCHMARK,
                TokenType.AUGMENT, TokenType.TOKENIZER, TokenType.CALLBACK,
                TokenType.RAG, TokenType.FEDERATED, TokenType.AUTOML,
                TokenType.PIPELINE, TokenType.PRETRAIN, TokenType.FINETUNE,
                TokenType.ENSEMBLE, TokenType.EXPLAIN, TokenType.SEARCH,
            }
            
            if key_token.type in keyword_key_types:
                key = self.advance().value
                if self.match(TokenType.COLON):
                    self.skip_newlines()
                    if self.peek().type == TokenType.INDENT:
                        result[key] = self._parse_sub_block()
                    else:
                        result[key] = self._parse_value()
                elif self.peek().type == TokenType.LPAREN:
                    self.pos -= 1
                    result[key] = self._parse_value()
                else:
                    result[key] = Identifier(name=key)
            elif key_token.type == TokenType.DASH:
                if "_list" not in result:
                    result["_list"] = []
                self.advance()
                result["_list"].append(self._parse_value())
            else:
                self.advance()
            
            self.skip_newlines()
        
        if self.peek().type == TokenType.DEDENT:
            self.advance()
        
        return result
    
    def _parse_sub_block(self) -> list:
        items = []
        if self.peek().type != TokenType.INDENT:
            return items
        self.advance()
        
        while not self.at_block_end():
            self.skip_newlines()
            if self.at_block_end():
                break
            
            token = self.peek()
            if token.type == TokenType.DASH:
                self.advance()
                items.append(self._parse_value())
            elif token.type in (TokenType.IDENTIFIER, TokenType.NUMBER) or token.type in (
                TokenType.MODEL, TokenType.DATA, TokenType.TRAIN, TokenType.EVALUATE,
                TokenType.TRANSFORM, TokenType.FORWARD, TokenType.DEPLOY, TokenType.SERVE,
                TokenType.TEST, TokenType.FEATURE, TokenType.GRAPH, TokenType.AUDIO,
                TokenType.AGENT, TokenType.METRIC, TokenType.EMBEDDING, TokenType.MONITOR,
                TokenType.GAN, TokenType.DIFFUSION, TokenType.RL, TokenType.TABULAR,
                TokenType.TIMESERIES, TokenType.MULTIMODAL, TokenType.DISTILL,
                TokenType.QUANTIZE, TokenType.BENCHMARK, TokenType.AUGMENT,
                TokenType.TOKENIZER, TokenType.CALLBACK, TokenType.RAG,
                TokenType.FEDERATED, TokenType.AUTOML, TokenType.PIPELINE,
                TokenType.PRETRAIN, TokenType.FINETUNE, TokenType.ENSEMBLE,
                TokenType.EXPLAIN, TokenType.SEARCH,
            ):
                val = self._parse_value()
                if self.peek().type == TokenType.COLON:
                    self.advance()
                    self.skip_newlines()
                    # Check for nested sub-block
                    if self.peek().type == TokenType.INDENT:
                        v = self._parse_sub_block()
                    else:
                        v = self._parse_value()
                    key_name = val.name if isinstance(val, Identifier) else str(val)
                    items.append(KeyValue(key=key_name, value=v))
                else:
                    items.append(val)
            else:
                items.append(self._parse_value())
            
            self.skip_newlines()
        
        if self.peek().type == TokenType.DEDENT:
            self.advance()
        return items
    
    # ─── Helper extractors ─────────────────────────────────────
    
    def _extract_list(self, block: dict, key: str) -> list:
        raw = block.get(key)
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, (Identifier, FunctionCall)):
            return [raw]
        return []
    
    def _extract_str(self, block: dict, key: str, default: str = "") -> str:
        val = block.get(key)
        if isinstance(val, Identifier):
            return val.name
        if isinstance(val, StringLiteral):
            return val.value
        if isinstance(val, str):
            return val
        return default
    
    def _extract_int(self, block: dict, key: str, default: int = 0) -> int:
        val = block.get(key)
        if isinstance(val, NumberLiteral):
            return int(val.value)
        if isinstance(val, (int, float)):
            return int(val)
        return default
    
    def _extract_float(self, block: dict, key: str, default: float = 0.0) -> float:
        val = block.get(key)
        if isinstance(val, NumberLiteral):
            return float(val.value)
        if isinstance(val, (int, float)):
            return float(val)
        return default
    
    def _extract_bool(self, block: dict, key: str, default: bool = False) -> bool:
        val = block.get(key)
        if isinstance(val, BoolLiteral):
            return val.value
        if isinstance(val, bool):
            return val
        return default
    
    def _remaining_config(self, block: dict, exclude: set) -> dict:
        return {k: v for k, v in block.items() if k not in exclude and k != "_list"}
    
    # ═══════════════════════════════════════════════════════════
    # CORE BLOCK PARSERS
    # ═══════════════════════════════════════════════════════════
    
    def _parse_model(self) -> ModelDef:
        name = self._parse_block_header(TokenType.MODEL)
        block = self._parse_key_value_block()
        model = ModelDef(name=name)
        for key, value in block.items():
            if key == "forward":
                model.forward_def = ForwardDef(body=[value] if not isinstance(value, list) else value)
            elif key in ("config", "device", "dtype"):
                model.config[key] = value
            elif key == "_list":
                continue
            else:
                model.layers[key] = value
        return model
    
    def _parse_data(self) -> DataDef:
        name = self._parse_block_header(TokenType.DATA)
        block = self._parse_key_value_block()
        data = DataDef(name=name)
        data.source = block.get("source")
        data.format = block.get("format")
        split_val = block.get("split")
        if isinstance(split_val, RatioExpr):
            data.split = split_val
        transforms = block.get("transform", block.get("transforms", []))
        if isinstance(transforms, list):
            data.transforms = [t for t in transforms if isinstance(t, FunctionCall)]
        loader = block.get("loader", {})
        if isinstance(loader, list):
            for item in loader:
                if isinstance(item, KeyValue):
                    data.loader_config[item.key] = item.value
        elif isinstance(loader, dict):
            data.loader_config = loader
        data.streaming = self._extract_bool(block, "streaming", False)
        data.cache = self._extract_bool(block, "cache", False)
        preprocessing = block.get("preprocessing", [])
        if isinstance(preprocessing, list):
            data.preprocessing = preprocessing
        data.config = self._remaining_config(block, {"source", "format", "split", "transform", "transforms", "loader", "streaming", "cache", "preprocessing"})
        return data
    
    def _parse_train(self) -> TrainDef:
        name = self._parse_block_header(TokenType.TRAIN)
        block = self._parse_key_value_block()
        train = TrainDef(name=name)
        train.model_ref = block.get("model")
        train.data_ref = block.get("data")
        train.optimizer = block.get("optimizer")
        train.scheduler = block.get("scheduler")
        train.loss = block.get("loss")
        train.epochs = self._extract_int(block, "epochs", 10)
        train.device = self._extract_str(block, "device", "auto")
        train.precision = self._extract_str(block, "precision", "fp32")
        train.callbacks = self._extract_list(block, "callbacks")
        train.metrics = self._extract_list(block, "metrics")
        # Advanced
        train.gradient_accumulation = self._extract_int(block, "gradient_accumulation", 1)
        train.gradient_clip = self._extract_float(block, "gradient_clip", 0.0) or None
        train.gradient_clip_norm = self._extract_float(block, "gradient_clip_norm", 0.0) or None
        train.ema = self._extract_bool(block, "ema", False)
        train.ema_decay = self._extract_float(block, "ema_decay", 0.999)
        train.swa = self._extract_bool(block, "swa", False)
        train.label_smoothing = self._extract_float(block, "label_smoothing", 0.0)
        train.mixup_alpha = self._extract_float(block, "mixup_alpha", 0.0)
        train.cutmix_alpha = self._extract_float(block, "cutmix_alpha", 0.0)
        train.compile_model = self._extract_bool(block, "compile", False)
        train.distributed = self._extract_str(block, "distributed", "") or None
        train.num_gpus = self._extract_int(block, "num_gpus", 1)
        train.num_nodes = self._extract_int(block, "num_nodes", 1)
        train.log_every = self._extract_int(block, "log_every", 10)
        train.eval_every = self._extract_int(block, "eval_every", 1)
        train.save_every = self._extract_int(block, "save_every", 1)
        excluded = {"model", "data", "optimizer", "scheduler", "loss", "epochs", "device",
                    "precision", "callbacks", "metrics", "gradient_accumulation", "gradient_clip",
                    "gradient_clip_norm", "ema", "ema_decay", "swa", "label_smoothing",
                    "mixup_alpha", "cutmix_alpha", "compile", "distributed", "num_gpus",
                    "num_nodes", "log_every", "eval_every", "save_every"}
        train.config = self._remaining_config(block, excluded)
        return train
    
    def _parse_evaluate(self) -> EvalDef:
        name = self._parse_block_header(TokenType.EVALUATE)
        block = self._parse_key_value_block()
        eval_def = EvalDef(name=name)
        eval_def.checkpoint = self._extract_str(block, "checkpoint", "best")
        eval_def.data_ref = block.get("data")
        eval_def.metrics = self._extract_list(block, "metrics")
        eval_def.export = self._extract_str(block, "export", "") or None
        eval_def.per_class = self._extract_bool(block, "per_class", False)
        return eval_def
    
    def _parse_search(self) -> SearchDef:
        name = self._parse_block_header(TokenType.SEARCH)
        block = self._parse_key_value_block()
        search = SearchDef(name=name)
        search.base_ref = self._extract_str(block, "base", "")
        search.method = self._extract_str(block, "method", "bayesian")
        search.trials = self._extract_int(block, "trials", 50)
        search.timeout = self._extract_int(block, "timeout", 0) or None
        search.parallel_trials = self._extract_int(block, "parallel_trials", 1)
        space = block.get("space", [])
        if isinstance(space, list):
            for item in space:
                if isinstance(item, KeyValue) and isinstance(item.value, FunctionCall):
                    search.space[item.key] = item.value
        search.objective = block.get("objective")
        search.pruner = block.get("pruner")
        return search
    
    def _parse_deploy(self) -> DeployDef:
        name = self._parse_block_header(TokenType.DEPLOY)
        block = self._parse_key_value_block()
        deploy = DeployDef(name=name)
        deploy.checkpoint = self._extract_str(block, "checkpoint", "best")
        deploy.format = self._extract_str(block, "format", "onnx")
        deploy.optimize = self._extract_bool(block, "optimize", False)
        deploy.quantize = self._extract_str(block, "quantize", "") or None
        deploy.prune = self._extract_str(block, "prune", "") or None
        deploy.docker = self._extract_bool(block, "docker", False)
        deploy.target_device = self._extract_str(block, "target_device", "") or None
        serve = block.get("serve", [])
        if isinstance(serve, list):
            for item in serve:
                if isinstance(item, KeyValue):
                    deploy.serve_config[item.key] = item.value
        return deploy
    
    def _parse_pipeline(self) -> PipelineDef:
        name = self._parse_block_header(TokenType.PIPELINE)
        block = self._parse_key_value_block()
        return PipelineDef(name=name, config=block, parallel=self._extract_bool(block, "parallel", False))
    
    def _parse_finetune(self) -> FinetuneDef:
        name = self._parse_block_header(TokenType.FINETUNE)
        block = self._parse_key_value_block()
        ft = FinetuneDef(name=name)
        ft.base_model = self._extract_str(block, "base_model", "") or self._extract_str(block, "base", "")
        ft.method = self._extract_str(block, "method", "full")
        ft.config = self._remaining_config(block, {"base_model", "base", "method"})
        return ft
    
    def _parse_ensemble(self) -> EnsembleDef:
        name = self._parse_block_header(TokenType.ENSEMBLE)
        block = self._parse_key_value_block()
        ens = EnsembleDef(name=name)
        models = block.get("models")
        if isinstance(models, ListLiteral):
            ens.models = [e.name if isinstance(e, Identifier) else str(e) for e in models.elements]
        ens.strategy = self._extract_str(block, "strategy", "voting")
        ens.meta_learner = self._extract_str(block, "meta_learner", "") or None
        return ens
    
    def _parse_explain(self) -> ExplainDef:
        name = self._parse_block_header(TokenType.EXPLAIN)
        block = self._parse_key_value_block()
        exp = ExplainDef(name=name)
        exp.model_ref = self._extract_str(block, "model", "")
        exp.method = self._extract_str(block, "method", "shap")
        exp.data_ref = block.get("data")
        exp.num_samples = self._extract_int(block, "num_samples", 100)
        return exp
    
    def _parse_pretrain(self) -> PretrainDef:
        name = self._parse_block_header(TokenType.PRETRAIN)
        block = self._parse_key_value_block()
        pt = PretrainDef(name=name)
        pt.model_ref = self._extract_str(block, "model", "")
        pt.objective = self._extract_str(block, "objective", "masked_lm")
        pt.data_ref = block.get("data")
        return pt
    
    # ═══════════════════════════════════════════════════════════
    # EXTENDED BLOCK PARSERS
    # ═══════════════════════════════════════════════════════════
    
    def _parse_gan(self) -> GANDef:
        name = self._parse_block_header(TokenType.GAN)
        block = self._parse_key_value_block()
        g = GANDef(name=name)
        g.generator = block.get("generator")
        g.discriminator = block.get("discriminator")
        g.latent_dim = self._extract_int(block, "latent_dim", 100)
        g.loss_type = self._extract_str(block, "loss", "vanilla") or self._extract_str(block, "loss_type", "vanilla")
        g.optimizer_g = block.get("optimizer_g")
        g.optimizer_d = block.get("optimizer_d")
        g.n_critic = self._extract_int(block, "n_critic", 1)
        g.gp_weight = self._extract_float(block, "gp_weight", 10.0)
        g.epochs = self._extract_int(block, "epochs", 100)
        g.config = self._remaining_config(block, {"generator", "discriminator", "latent_dim", "loss", "loss_type", "optimizer_g", "optimizer_d", "n_critic", "gp_weight", "epochs"})
        return g
    
    def _parse_diffusion(self) -> DiffusionDef:
        name = self._parse_block_header(TokenType.DIFFUSION)
        block = self._parse_key_value_block()
        d = DiffusionDef(name=name)
        d.model_ref = block.get("model")
        d.noise_scheduler = self._extract_str(block, "scheduler", "ddpm") or self._extract_str(block, "noise_scheduler", "ddpm")
        d.timesteps = self._extract_int(block, "timesteps", 1000)
        d.beta_schedule = self._extract_str(block, "beta_schedule", "linear")
        d.image_size = self._extract_int(block, "image_size", 256)
        d.channels = self._extract_int(block, "channels", 3)
        d.guidance_scale = self._extract_float(block, "guidance_scale", 7.5)
        d.config = self._remaining_config(block, {"model", "scheduler", "noise_scheduler", "timesteps", "beta_schedule", "image_size", "channels", "guidance_scale"})
        return d
    
    def _parse_rl(self) -> RLDef:
        name = self._parse_block_header(TokenType.RL)
        block = self._parse_key_value_block()
        r = RLDef(name=name)
        r.algorithm = self._extract_str(block, "algorithm", "ppo")
        r.environment = self._extract_str(block, "environment", "")
        r.policy = self._extract_str(block, "policy", "MlpPolicy")
        r.total_timesteps = self._extract_int(block, "total_timesteps", 100000)
        r.learning_rate = self._extract_float(block, "learning_rate", 3e-4)
        r.gamma = self._extract_float(block, "gamma", 0.99)
        r.n_steps = self._extract_int(block, "n_steps", 2048)
        r.batch_size = self._extract_int(block, "batch_size", 64)
        r.n_epochs = self._extract_int(block, "n_epochs", 10)
        r.clip_range = self._extract_float(block, "clip_range", 0.2)
        r.buffer_size = self._extract_int(block, "buffer_size", 100000)
        r.config = self._remaining_config(block, {"algorithm", "environment", "policy", "total_timesteps", "learning_rate", "gamma", "n_steps", "batch_size", "n_epochs", "clip_range", "buffer_size"})
        return r
    
    def _parse_tabular(self) -> TabularDef:
        name = self._parse_block_header(TokenType.TABULAR)
        block = self._parse_key_value_block()
        t = TabularDef(name=name)
        t.task = self._extract_str(block, "task", "classification")
        t.algorithm = self._extract_str(block, "algorithm", "xgboost")
        t.data_ref = block.get("data")
        t.target_column = self._extract_str(block, "target", "") or self._extract_str(block, "target_column", "")
        t.cross_validation = self._extract_int(block, "cross_validation", 5) or self._extract_int(block, "cv", 5)
        feature_cols = block.get("feature_columns", block.get("features", []))
        if isinstance(feature_cols, ListLiteral):
            t.feature_columns = [e.name if isinstance(e, Identifier) else str(e) for e in feature_cols.elements]
        t.config = self._remaining_config(block, {"task", "algorithm", "data", "target", "target_column", "cross_validation", "cv", "feature_columns", "features"})
        return t
    
    def _parse_timeseries(self) -> TimeSeriesDef:
        name = self._parse_block_header(TokenType.TIMESERIES)
        block = self._parse_key_value_block()
        ts = TimeSeriesDef(name=name)
        ts.task = self._extract_str(block, "task", "forecast")
        ts.algorithm = self._extract_str(block, "algorithm", "transformer")
        ts.data_ref = block.get("data")
        ts.target_column = self._extract_str(block, "target", "") or self._extract_str(block, "target_column", "")
        ts.time_column = self._extract_str(block, "time_column", "") or self._extract_str(block, "time", "")
        ts.horizon = self._extract_int(block, "horizon", 24)
        ts.lookback = self._extract_int(block, "lookback", 168)
        ts.frequency = self._extract_str(block, "frequency", "1h")
        ts.config = self._remaining_config(block, {"task", "algorithm", "data", "target", "target_column", "time_column", "time", "horizon", "lookback", "frequency"})
        return ts
    
    def _parse_graph(self) -> GraphDef:
        name = self._parse_block_header(TokenType.GRAPH)
        block = self._parse_key_value_block()
        g = GraphDef(name=name)
        g.task = self._extract_str(block, "task", "node_classification")
        g.conv_type = self._extract_str(block, "conv_type", "GCN") or self._extract_str(block, "conv", "GCN")
        g.num_features = self._extract_int(block, "num_features", 0) or self._extract_int(block, "in_features", 0)
        g.num_classes = self._extract_int(block, "num_classes", 0) or self._extract_int(block, "out_features", 0)
        g.hidden_dim = self._extract_int(block, "hidden_dim", 64)
        g.heads = self._extract_int(block, "heads", 1)
        g.dropout = self._extract_float(block, "dropout", 0.5)
        g.pooling = self._extract_str(block, "pooling", "mean")
        g.config = self._remaining_config(block, {"task", "conv_type", "conv", "num_features", "in_features", "num_classes", "out_features", "hidden_dim", "heads", "dropout", "pooling"})
        return g
    
    def _parse_audio(self) -> AudioDef:
        name = self._parse_block_header(TokenType.AUDIO)
        block = self._parse_key_value_block()
        a = AudioDef(name=name)
        a.task = self._extract_str(block, "task", "classification")
        a.model_type = self._extract_str(block, "model", "") or self._extract_str(block, "model_type", "wav2vec2")
        a.sample_rate = self._extract_int(block, "sample_rate", 16000)
        a.n_mels = self._extract_int(block, "n_mels", 80)
        a.n_fft = self._extract_int(block, "n_fft", 1024)
        a.hop_length = self._extract_int(block, "hop_length", 512)
        a.data_ref = block.get("data")
        a.config = self._remaining_config(block, {"task", "model", "model_type", "sample_rate", "n_mels", "n_fft", "hop_length", "data"})
        return a
    
    def _parse_multimodal(self) -> MultimodalDef:
        name = self._parse_block_header(TokenType.MULTIMODAL)
        block = self._parse_key_value_block()
        m = MultimodalDef(name=name)
        m.task = self._extract_str(block, "task", "vqa")
        modalities = block.get("modalities")
        if isinstance(modalities, ListLiteral):
            m.modalities = [e.name if isinstance(e, Identifier) else e.value if isinstance(e, StringLiteral) else str(e) for e in modalities.elements]
        m.vision_encoder = self._extract_str(block, "vision_encoder", "") or None
        m.text_encoder = self._extract_str(block, "text_encoder", "") or None
        m.audio_encoder = self._extract_str(block, "audio_encoder", "") or None
        m.fusion_method = self._extract_str(block, "fusion", "") or self._extract_str(block, "fusion_method", "cross_attention")
        m.config = self._remaining_config(block, {"task", "modalities", "vision_encoder", "text_encoder", "audio_encoder", "fusion", "fusion_method"})
        return m
    
    def _parse_distill(self) -> DistillDef:
        name = self._parse_block_header(TokenType.DISTILL)
        block = self._parse_key_value_block()
        d = DistillDef(name=name)
        d.teacher = self._extract_str(block, "teacher", "")
        d.student = self._extract_str(block, "student", "")
        d.method = self._extract_str(block, "method", "kd")
        d.temperature = self._extract_float(block, "temperature", 4.0)
        d.alpha = self._extract_float(block, "alpha", 0.7)
        d.data_ref = block.get("data")
        d.config = self._remaining_config(block, {"teacher", "student", "method", "temperature", "alpha", "data"})
        return d
    
    def _parse_quantize(self) -> QuantizeDef:
        name = self._parse_block_header(TokenType.QUANTIZE)
        block = self._parse_key_value_block()
        q = QuantizeDef(name=name)
        q.model_ref = self._extract_str(block, "model", "")
        q.method = self._extract_str(block, "method", "dynamic")
        q.dtype = self._extract_str(block, "dtype", "int8")
        q.calibration_data = self._extract_str(block, "calibration_data", "") or None
        q.config = self._remaining_config(block, {"model", "method", "dtype", "calibration_data"})
        return q
    
    def _parse_monitor(self) -> MonitorDef:
        name = self._parse_block_header(TokenType.MONITOR)
        block = self._parse_key_value_block()
        m = MonitorDef(name=name)
        m.model_ref = self._extract_str(block, "model", "")
        m.backend = self._extract_str(block, "backend", "wandb")
        m.metrics = self._extract_list(block, "metrics")
        m.alerts = self._extract_list(block, "alerts")
        m.drift_detection = self._extract_bool(block, "drift_detection", False)
        m.config = self._remaining_config(block, {"model", "backend", "metrics", "alerts", "drift_detection"})
        return m
    
    def _parse_serve(self) -> ServeDef:
        name = self._parse_block_header(TokenType.SERVE)
        block = self._parse_key_value_block()
        s = ServeDef(name=name)
        s.model_ref = self._extract_str(block, "model", "")
        s.framework = self._extract_str(block, "framework", "fastapi")
        s.endpoint = self._extract_str(block, "endpoint", "/predict")
        s.host = self._extract_str(block, "host", "0.0.0.0")
        s.port = self._extract_int(block, "port", 8000)
        s.batch = self._extract_bool(block, "batch", False)
        s.max_batch_size = self._extract_int(block, "max_batch_size", 32)
        s.timeout = self._extract_int(block, "timeout", 30)
        s.cors = self._extract_bool(block, "cors", True)
        s.config = self._remaining_config(block, {"model", "framework", "endpoint", "host", "port", "batch", "max_batch_size", "timeout", "cors"})
        return s
    
    def _parse_test(self) -> TestDef:
        name = self._parse_block_header(TokenType.TEST)
        block = self._parse_key_value_block()
        t = TestDef(name=name)
        t.model_ref = self._extract_str(block, "model", "")
        t.tests = self._extract_list(block, "tests")
        t.data_ref = block.get("data")
        t.config = self._remaining_config(block, {"model", "tests", "data"})
        return t
    
    def _parse_benchmark(self) -> BenchmarkDef:
        name = self._parse_block_header(TokenType.BENCHMARK)
        block = self._parse_key_value_block()
        b = BenchmarkDef(name=name)
        b.model_ref = self._extract_str(block, "model", "")
        b.metrics = self._extract_list(block, "metrics")
        b.num_warmup = self._extract_int(block, "num_warmup", 10)
        b.num_runs = self._extract_int(block, "num_runs", 100)
        b.config = self._remaining_config(block, {"model", "metrics", "num_warmup", "num_runs"})
        return b
    
    def _parse_augment(self) -> AugmentDef:
        name = self._parse_block_header(TokenType.AUGMENT)
        block = self._parse_key_value_block()
        a = AugmentDef(name=name)
        a.domain = self._extract_str(block, "domain", "image")
        a.transforms = self._extract_list(block, "transforms") or self._extract_list(block, "_list") or block.get("_list", [])
        a.probability = self._extract_float(block, "probability", 1.0)
        a.config = self._remaining_config(block, {"domain", "transforms", "probability"})
        return a
    
    def _parse_feature(self) -> FeatureDef:
        name = self._parse_block_header(TokenType.FEATURE)
        block = self._parse_key_value_block()
        f = FeatureDef(name=name)
        f.data_ref = block.get("data")
        f.operations = self._extract_list(block, "operations") or self._extract_list(block, "_list") or block.get("_list", [])
        f.target = self._extract_str(block, "target", "") or None
        f.config = self._remaining_config(block, {"data", "operations", "target"})
        return f
    
    def _parse_embedding(self) -> EmbeddingDef:
        name = self._parse_block_header(TokenType.EMBEDDING)
        block = self._parse_key_value_block()
        e = EmbeddingDef(name=name)
        e.model = self._extract_str(block, "model", "")
        e.dim = self._extract_int(block, "dim", 768)
        e.pooling = self._extract_str(block, "pooling", "mean")
        e.normalize = self._extract_bool(block, "normalize", True)
        e.config = self._remaining_config(block, {"model", "dim", "pooling", "normalize"})
        return e
    
    def _parse_tokenizer(self) -> TokenizerDef:
        name = self._parse_block_header(TokenType.TOKENIZER)
        block = self._parse_key_value_block()
        t = TokenizerDef(name=name)
        t.type = self._extract_str(block, "type", "bpe")
        t.vocab_size = self._extract_int(block, "vocab_size", 32000)
        t.max_length = self._extract_int(block, "max_length", 512)
        t.padding = self._extract_str(block, "padding", "max_length")
        t.truncation = self._extract_bool(block, "truncation", True)
        t.config = self._remaining_config(block, {"type", "vocab_size", "max_length", "padding", "truncation"})
        return t
    
    def _parse_callback(self) -> CallbackDef:
        name = self._parse_block_header(TokenType.CALLBACK)
        block = self._parse_key_value_block()
        c = CallbackDef(name=name)
        c.trigger = self._extract_str(block, "trigger", "epoch_end")
        c.actions = self._extract_list(block, "actions") or self._extract_list(block, "_list") or block.get("_list", [])
        c.config = self._remaining_config(block, {"trigger", "actions"})
        return c
    
    def _parse_metric(self) -> MetricDef:
        name = self._parse_block_header(TokenType.METRIC)
        block = self._parse_key_value_block()
        m = MetricDef(name=name)
        m.formula = self._extract_str(block, "formula", "") or None
        m.higher_is_better = self._extract_bool(block, "higher_is_better", True)
        m.config = self._remaining_config(block, {"formula", "higher_is_better"})
        return m
    
    def _parse_rag(self) -> RAGDef:
        name = self._parse_block_header(TokenType.RAG)
        block = self._parse_key_value_block()
        r = RAGDef(name=name)
        r.retriever = self._extract_str(block, "retriever", "faiss")
        r.generator = self._extract_str(block, "generator", "")
        r.embedding_model = self._extract_str(block, "embedding_model", "") or self._extract_str(block, "embedding", "")
        r.chunk_size = self._extract_int(block, "chunk_size", 512)
        r.chunk_overlap = self._extract_int(block, "chunk_overlap", 50)
        r.top_k = self._extract_int(block, "top_k", 5)
        r.config = self._remaining_config(block, {"retriever", "generator", "embedding_model", "embedding", "chunk_size", "chunk_overlap", "top_k"})
        return r
    
    def _parse_agent(self) -> AgentDef:
        name = self._parse_block_header(TokenType.AGENT)
        block = self._parse_key_value_block()
        a = AgentDef(name=name)
        a.llm = self._extract_str(block, "llm", "")
        tools = block.get("tools")
        if isinstance(tools, ListLiteral):
            a.tools = [e.name if isinstance(e, Identifier) else e.value if isinstance(e, StringLiteral) else str(e) for e in tools.elements]
        elif isinstance(tools, list):
            a.tools = tools
        a.memory = self._extract_str(block, "memory", "") or None
        a.max_iterations = self._extract_int(block, "max_iterations", 10)
        a.system_prompt = self._extract_str(block, "system_prompt", "") or None
        a.config = self._remaining_config(block, {"llm", "tools", "memory", "max_iterations", "system_prompt"})
        return a
    
    def _parse_federated(self) -> FederatedDef:
        name = self._parse_block_header(TokenType.FEDERATED)
        block = self._parse_key_value_block()
        f = FederatedDef(name=name)
        f.model_ref = self._extract_str(block, "model", "")
        f.num_clients = self._extract_int(block, "num_clients", 10)
        f.rounds = self._extract_int(block, "rounds", 100)
        f.strategy = self._extract_str(block, "strategy", "fedavg")
        f.fraction_fit = self._extract_float(block, "fraction_fit", 0.5)
        f.config = self._remaining_config(block, {"model", "num_clients", "rounds", "strategy", "fraction_fit"})
        return f
    
    def _parse_automl(self) -> AutoMLDef:
        name = self._parse_block_header(TokenType.AUTOML)
        block = self._parse_key_value_block()
        a = AutoMLDef(name=name)
        a.task = self._extract_str(block, "task", "classification")
        a.data_ref = block.get("data")
        a.target_column = self._extract_str(block, "target", "") or self._extract_str(block, "target_column", "")
        a.time_budget = self._extract_int(block, "time_budget", 3600)
        a.metric = self._extract_str(block, "metric", "accuracy")
        a.framework = self._extract_str(block, "framework", "auto")
        a.config = self._remaining_config(block, {"task", "data", "target", "target_column", "time_budget", "metric", "framework"})
        return a
    
    # ─── Pass-through ──────────────────────────────────────────
    
    def _parse_python_block(self) -> PythonBlock:
        token = self.advance()
        code = token.value.replace("@python:", "").strip()
        return PythonBlock(code=code)
    
    def _parse_import(self) -> PythonBlock:
        tokens = []
        while self.peek().type != TokenType.NEWLINE and self.peek().type != TokenType.EOF:
            tokens.append(self.advance().value)
        return PythonBlock(code=" ".join(tokens))
