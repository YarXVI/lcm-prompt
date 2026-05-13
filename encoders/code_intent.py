import re
from typing import List

from ..ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION
from ..encoder_base import ContentEncoder, EncodingContext
from ..chunk_store import Chunk


class CodeIntentEncoder(ContentEncoder):

    _DEF_PATTERNS = [
        re.compile(r"def\s+(\w+)"),
        re.compile(r"class\s+(\w+)"),
        re.compile(r"async\s+def\s+(\w+)"),
        re.compile(r"func\s+(\w+)"),
        re.compile(r"func\s+\(\w+\s+\*?\w+\)\s+(\w+)"),
        re.compile(r"fn\s+(\w+)"),
        re.compile(r"pub\s+(?:async\s+)?fn\s+(\w+)"),
        re.compile(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]"),
        re.compile(r"(?:export\s+)?function\s+(\w+)"),
        re.compile(r"(?:export\s+)?interface\s+(\w+)"),
        re.compile(r"(?:export\s+)?type\s+(\w+)"),
        re.compile(r"(?:export\s+)?enum\s+(\w+)"),
        re.compile(r"type\s+(\w+)\s+struct"),
        re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)"),
    ]

    _CALL_PATTERN = re.compile(r"(\w+)\(")

    _SIG_PREFIXES = (
        "def ", "class ", "async def ", "func ", "fn ", "pub ",
        "interface ", "type ", "enum ", "struct ", "impl ",
        "const ", "let ", "var ", "export ", "function ",
        "CREATE TABLE ", "CREATE INDEX ", "CREATE OR REPLACE ",
        "ALTER TABLE ", "DROP TABLE ",
    )

    _CODE_INDICATORS = re.compile(
        r'(?:^|\n)\s*'
        r'(?:'
        r'def |class |async def |func |fn |pub |'
        r'const |let |var |function |export |'
        r'interface |type |enum |struct |impl |'
        r'import |from |require\(|use |package |'
        r'return |if \(|for \(|while \(|switch \(|case |'
        r'try {|catch |finally {|throw |raise |'
        r'=> \{|=> \(|=\> [a-zA-Z]|'
        r'public |private |protected |static |'
        r'@\w+|'
        r'CREATE TABLE |CREATE INDEX |CREATE OR REPLACE |ALTER TABLE |DROP TABLE |'
        r'SELECT .* FROM |INSERT INTO |UPDATE .* SET |DELETE FROM |'
        r'JOIN |INNER JOIN |LEFT JOIN |WHERE |GROUP BY |ORDER BY |HAVING |'
        r'REFERENCES |PRIMARY KEY |FOREIGN KEY |UNIQUE |CHECK |DEFAULT |'
        r'CREATE FUNCTION |CREATE TRIGGER |CREATE VIEW |'
        r'\w+:$|'
        r'-{2,3} |'
        r'\{\{.*\}\}|'
        r'\S+: \S+'
        r')'
    )

    @property
    def encoding_type(self) -> str:
        return "code-intent"

    @property
    def supported_languages(self) -> List[str]:
        return ["code", "python", "javascript", "typescript", "go", "rust", "java"]

    def detect(self, text: str) -> float:
        indicator_count = len(self._CODE_INDICATORS.findall(text))
        total_lines = max(len(text.split("\n")), 1)
        ratio = indicator_count / total_lines
        return min(ratio * 4.0, 1.0)

    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        keywords = self._extract_keywords(text)
        summary = self._extract_summary(text)
        detail = self._extract_detail(text)
        full_tokens = Chunk._estimate_tokens(text)

        return MultiGranularityIR(
            encoding_type=self.encoding_type,
            source_language="code",
            grains={
                GrainLevel.KEYWORDS: Grain(
                    GrainLevel.KEYWORDS, keywords,
                    Chunk._estimate_tokens(keywords), reversible=False,
                ),
                GrainLevel.SUMMARY: Grain(
                    GrainLevel.SUMMARY, summary,
                    Chunk._estimate_tokens(summary), reversible=False,
                ),
                GrainLevel.DETAIL: Grain(
                    GrainLevel.DETAIL, detail,
                    Chunk._estimate_tokens(detail), reversible=False,
                ),
                GrainLevel.FULL: Grain(
                    GrainLevel.FULL, text, full_tokens, reversible=True,
                ),
            },
            structure={"call_graph": self._extract_calls(text)},
            version=IR_VERSION,
        )

    def _extract_keywords(self, text: str) -> str:
        names = set()
        for pat in self._DEF_PATTERNS:
            for m in pat.finditer(text):
                names.add(m.group(1))
        if not names:
            identifiers = re.findall(r'\b([A-Z][a-zA-Z]+)\b', text)
            names = set(identifiers[:20])
        return ", ".join(sorted(names)[:20])

    def _extract_summary(self, text: str) -> str:
        docstrings = re.findall(r'"""(.*?)"""', text, re.DOTALL)
        if docstrings:
            first = docstrings[0].strip().split("\n")[0]
            return first[:300]
        docstrings2 = re.findall(r"'''(.*?)'''", text, re.DOTALL)
        if docstrings2:
            first = docstrings2[0].strip().split("\n")[0]
            return first[:300]
        single_comments = re.findall(r'//\s*(.+)', text)
        if single_comments:
            return single_comments[0][:300]
        names = self._extract_keywords(text)
        return f"Module: {names}" if names else "Code module"

    def _extract_detail(self, text: str) -> str:
        sig_lines = []
        comment_lines = []
        for line in text.split("\n"):
            s = line.strip()
            if any(s.startswith(p) for p in self._SIG_PREFIXES):
                sig_lines.append(s)
            elif s.startswith("#") or s.startswith("//"):
                comment_lines.append(s)

        result = sig_lines[:40]
        comment_budget = max(10, 50 - len(result))
        result.extend(comment_lines[:comment_budget])
        return "\n".join(result[:50])

    def _extract_calls(self, text: str) -> List[str]:
        calls = []
        for m in self._CALL_PATTERN.finditer(text):
            name = m.group(1)
            if not name[0].isupper() and name not in (
                "if", "for", "while", "with", "return", "print",
                "len", "range", "str", "int", "list", "dict", "set",
                "type", "isinstance", "hasattr", "getattr", "super",
                "require", "console",
            ):
                calls.append(name)
        return list(dict.fromkeys(calls))[:30]
