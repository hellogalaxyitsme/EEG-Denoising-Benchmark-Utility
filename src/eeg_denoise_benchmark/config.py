"""Small config helpers for flat YAML/JSON experiment files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    """Parse a simple scalar from a flat config file."""

    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or a deliberately simple flat YAML file."""

    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        config[key.strip().replace("-", "_")] = parse_scalar(value)
    return config


def dump_json(path: str | Path, data: Any) -> None:
    """Write stable, pretty JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
