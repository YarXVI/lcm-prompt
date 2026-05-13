import re
from typing import List, Dict, Any, Optional, Set

from ..ir_models import GrainLevel, Grain, MultiGranularityIR, IR_VERSION
from ..encoder_base import ContentEncoder, EncodingContext

try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_go as tsgo
    import tree_sitter_rust as tsrust
    import tree_sitter_typescript as tstypescript
    from tree_sitter import Language, Parser

    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

from .code_intent import CodeIntentEncoder


_LANG_MAP = {}
if _TS_AVAILABLE:
    _LANG_MAP = {
        "python": Language(tspython.language()),
        "javascript": Language(tsjavascript.language()),
        "go": Language(tsgo.language()),
        "rust": Language(tsrust.language()),
        "typescript": Language(tstypescript.language_typescript()),
        "tsx": Language(tstypescript.language_tsx()),
    }

_DEF_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "interface_declaration", "type_alias_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "interface_declaration", "method_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "enum_item", "impl_item", "trait_item"},
}

_CALL_NODE_TYPES = {
    "python": "call",
    "javascript": "call_expression",
    "typescript": "call_expression",
    "tsx": "call_expression",
    "go": "call_expression",
    "rust": "call_expression",
}

_IMPORT_NODE_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "go": {"import_declaration"},
    "rust": {"use_declaration"},
}


def _detect_language(text: str) -> Optional[str]:
    if "func " in text and "package " in text:
        return "go"
    if "fn " in text and ("impl " in text or "pub " in text or "let " in text):
        return "rust"
    if "def " in text and ("import " in text or "from " in text or "class " in text):
        return "python"
    if "def " in text:
        return "python"
    has_export = "export " in text
    has_import_from = bool(re.search(r'import\s+\w', text))
    has_class = "class " in text
    has_function = "function " in text
    has_const_let_var = bool(re.search(r"(?:const|let|var)\s+\w+\s*[:=]", text))
    if (has_export or has_import_from) and (has_function or has_class or has_const_let_var):
        if re.search(r"<React|useState|useEffect|\.tsx", text):
            return "tsx"
        return "javascript"
    if re.search(r"interface\s+\w+|type\s+\w+\s*=", text) and ":" in text:
        return "typescript"
    return None


class ASTCodeEncoder(ContentEncoder):

    _FALLBACK = CodeIntentEncoder()

    @property
    def encoding_type(self) -> str:
        return "code-intent-ast"

    @property
    def supported_languages(self) -> List[str]:
        return ["python", "javascript", "typescript", "tsx", "go", "rust", "code"]

    @property
    def ir_version(self) -> int:
        return IR_VERSION

    def detect(self, text: str) -> float:
        if not _TS_AVAILABLE:
            return 0.0
        lang = _detect_language(text)
        if lang and lang in _LANG_MAP:
            return min(self._FALLBACK.detect(text) * 1.1, 1.0)
        return 0.0

    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        if not _TS_AVAILABLE:
            return self._FALLBACK.encode(text, context)

        lang = _detect_language(text)
        if not lang or lang not in _LANG_MAP:
            return self._FALLBACK.encode(text, context)

        tree = self._parse(text, lang)
        if tree is None:
            return self._FALLBACK.encode(text, context)

        definitions = self._extract_definitions(tree, text, lang)
        call_graph = self._extract_call_graph(tree, text, lang)
        docstrings = self._extract_docstrings(tree, text, lang)
        imports = self._extract_imports(tree, text, lang)

        keywords = self._build_keywords(definitions, imports)
        summary = self._build_summary(definitions, docstrings, lang)
        detail = self._build_detail(definitions, text, lang)
        full_tokens = self._estimate_tokens(text)

        return MultiGranularityIR(
            encoding_type=self.encoding_type,
            source_language=lang,
            grains={
                GrainLevel.KEYWORDS: Grain(GrainLevel.KEYWORDS, keywords, self._estimate_tokens(keywords), False),
                GrainLevel.SUMMARY: Grain(GrainLevel.SUMMARY, summary, self._estimate_tokens(summary), False),
                GrainLevel.DETAIL: Grain(GrainLevel.DETAIL, detail, self._estimate_tokens(detail), False),
                GrainLevel.FULL: Grain(GrainLevel.FULL, text, full_tokens, True),
            },
            structure={
                "call_graph": call_graph,
                "definitions": [{"name": d["name"], "kind": d["kind"], "line": d.get("line", 0)} for d in definitions],
                "imports": imports,
                "language": lang,
            },
        )

    def _parse(self, text: str, lang: str):
        try:
            parser = Parser(_LANG_MAP[lang])
            tree = parser.parse(text.encode("utf-8"))
            return tree
        except Exception:
            return None

    def _extract_definitions(self, tree, text: str, lang: str) -> List[Dict[str, Any]]:
        defs = []
        source = text.encode("utf-8")
        root = tree.root_node
        def_types = _DEF_NODE_TYPES.get(lang, set())

        def visit(node):
            if node.type in def_types:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    for child in node.children:
                        if child.type == "identifier" or child.type == "type_identifier" or child.type == "field_identifier":
                            name_node = child
                            break

                name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace") if name_node else "unknown"
                sig = source[node.start_byte:min(node.end_byte, node.start_byte + 200)].decode("utf-8", errors="replace")
                first_line = sig.split("\n")[0]

                defs.append({
                    "name": name,
                    "kind": node.type,
                    "line": node.start_point[0] + 1,
                    "signature": first_line[:120],
                })

            for child in node.children:
                visit(child)

        visit(root)

        seen = set()
        unique = []
        for d in defs:
            key = (d["name"], d["kind"])
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def _extract_call_graph(self, tree, text: str, lang: str) -> List[str]:
        calls = []
        source = text.encode("utf-8")
        root = tree.root_node
        call_type = _CALL_NODE_TYPES.get(lang)

        if not call_type:
            return []

        def visit(node):
            if node.type == call_type:
                func_node = node.child_by_field_name("function")
                if func_node is None:
                    for child in node.children:
                        if child.type == "identifier":
                            func_node = child
                            break

                if func_node:
                    name = source[func_node.start_byte:func_node.end_byte].decode("utf-8", errors="replace")
                    if name not in calls:
                        calls.append(name)

            for child in node.children:
                visit(child)

        visit(root)
        return calls[:30]

    def _extract_docstrings(self, tree, text: str, lang: str) -> List[str]:
        docs = []
        source = text.encode("utf-8")
        root = tree.root_node

        if lang == "python":
            def visit(node):
                if node.type == "expression_statement":
                    for child in node.children:
                        if child.type == "string":
                            doc_text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                            cleaned = doc_text.strip('"""').strip("'''").strip('"').strip("'")
                            if len(cleaned) > 10:
                                docs.append(cleaned[:200])
                for child in node.children:
                    visit(child)
            visit(root)
        else:
            def visit_comment(node):
                if node.type == "comment":
                    comment = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                    if len(comment.strip()) > 5:
                        docs.append(comment.strip()[:200])
                for child in node.children:
                    visit_comment(child)
            visit_comment(root)

        return docs[:10]

    def _extract_imports(self, tree, text: str, lang: str) -> List[str]:
        imports = []
        source = text.encode("utf-8")
        root = tree.root_node
        import_types = _IMPORT_NODE_TYPES.get(lang, set())

        def visit(node):
            if node.type in import_types:
                imp_text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                imports.append(imp_text.strip()[:100])
            for child in node.children:
                visit(child)

        visit(root)
        return imports[:15]

    def _build_keywords(self, definitions: List[Dict], imports: List[str]) -> str:
        names = [d["name"] for d in definitions[:20]]
        return ", ".join(names)

    def _build_summary(self, definitions: List[Dict], docstrings: List[str], lang: str) -> str:
        if docstrings:
            return docstrings[0][:300]

        class_names = [d["name"] for d in definitions if "class" in d["kind"] or "struct" in d["kind"] or "impl" in d["kind"]]
        func_names = [d["name"] for d in definitions if "function" in d["kind"] or "method" in d["kind"] or "fn" in d["kind"].lower()]

        parts = []
        if class_names:
            parts.append(f"Classes: {', '.join(class_names[:5])}")
        if func_names:
            parts.append(f"Functions: {', '.join(func_names[:8])}")
        if not parts:
            all_names = [d["name"] for d in definitions[:10]]
            parts.append(f"Module: {', '.join(all_names)}")

        return "; ".join(parts)

    def _build_detail(self, definitions: List[Dict], text: str, lang: str) -> str:
        sigs = [d["signature"] for d in definitions[:40]]
        return "\n".join(sigs)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = len(text) - zh
        return zh * 2 + en // 4
