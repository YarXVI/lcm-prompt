"""
Environment Variable Loader
Loads .env files into os.environ with support for ${env:VAR} syntax.
"""

import os
import re
from pathlib import Path


_ENV_VAR_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env(path: str = "", override: bool = False) -> dict:
    """
    Load environment variables from a .env file.

    Supports `${env:VAR_NAME}` syntax to inherit from existing environment variables.

    Args:
        path: Path to .env file. Defaults to '.env' in the current directory.
        override: If True, override existing environment variables.

    Returns:
        Dict of loaded key-value pairs.
    """
    if not path:
        path = str(Path.cwd() / ".env")

    if not os.path.exists(path):
        return {}

    loaded = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            value = _resolve_env_vars(value)

            if not override and key in os.environ:
                continue

            os.environ[key] = value
            loaded[key] = value

    return loaded


def _resolve_env_vars(value: str) -> str:
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return _ENV_VAR_PATTERN.sub(_replace, value)
