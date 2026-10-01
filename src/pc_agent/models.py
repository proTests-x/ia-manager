from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ollama import OllamaClient


@dataclass(slots=True)
class ModelRecord:
    name: str
    backend: str = "ollama"
    local: bool = True
    size_bytes: int = 0
    parameter_billions: float = 0.0
    quantization: str = "unknown"
    architecture: str = "unknown"
    context_max: int = 8192
    embedding_length: int = 0
    layers: int = 0
    kv_bytes_per_token: int = 0
    capabilities: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=lambda: ["text"])
    tools: bool = False
    vision: bool = False
    reasoning: bool = False
    programming_score: float = 0.5
    structured_output: bool = False
    languages: list[str] = field(default_factory=lambda: ["multilingual"])
    speed_tokens_second: float | None = None
    ram_required_bytes: int = 0
    vram_required_bytes: int = 0
    backend_required: str = "Ollama"
    remote_model_id: str | None = None
    cost_input_per_million: float = 0.0
    cost_output_per_million: float = 0.0
    verified: dict[str, bool] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def has(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))


def _first_info(info: dict[str, Any], suffix: str, default: int = 0) -> int:
    for key, value in info.items():
        if key.endswith(suffix):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def _raw_info(info: dict[str, Any], suffix: str) -> Any:
    for key, value in info.items():
        if key.endswith(suffix):
            return value
    return None


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, ModelRecord] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw.get("models", []):
                record = ModelRecord(**item)
                self.records[record.name] = record
        except (OSError, ValueError, TypeError):
            self.records = {}

    def save(self) -> None:
        payload = {"models": [asdict(record) for record in self.records.values()]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def refresh_ollama(self, client: OllamaClient, tags: dict[str, Any]) -> list[ModelRecord]:
        prior_records = dict(self.records)
        self._prior_records = prior_records
        refreshed: dict[str, ModelRecord] = {}
        for tag in tags.get("models", []):
            if not isinstance(tag, dict) or not tag.get("name"):
                continue
            name = str(tag["name"])
            show = client.show(name)
            details = show.get("details") or {}
            info = show.get("model_info") or {}
            capabilities = [str(value) for value in show.get("capabilities") or []]
            params_text = str(details.get("parameter_size") or "0")
            match = re.search(r"([0-9.]+)\s*B", params_text, re.IGNORECASE)
            params = float(match.group(1)) if match else 0.0
            context = _first_info(info, ".context_length", 8192)
            embedding = _first_info(info, ".embedding_length")
            layers = _first_info(info, ".block_count")
            kv_heads_raw = _raw_info(info, ".attention.head_count_kv")
            key_length = _first_info(info, ".attention.key_length")
            value_length = _first_info(info, ".attention.value_length")
            if isinstance(kv_heads_raw, list):
                kv_head_layers = sum(int(value or 0) for value in kv_heads_raw)
            else:
                kv_head_layers = layers * int(kv_heads_raw or 0)
            kv_bytes = kv_head_layers * (key_length + value_length) * 2
            size = int(tag.get("size") or 0)
            lower = name.lower()
            programming = 0.9 if any(word in lower for word in ("code", "coder")) else min(0.9, 0.48 + params / 80)
            previous = prior_records.get(name)
            record = ModelRecord(
                name=name,
                size_bytes=size,
                parameter_billions=params,
                quantization=str(details.get("quantization_level") or "unknown"),
                architecture=str(details.get("family") or "unknown"),
                context_max=context,
                embedding_length=embedding,
                layers=layers,
                kv_bytes_per_token=kv_bytes,
                capabilities=capabilities,
                modalities=["text", "image"] if "vision" in capabilities else ["text"],
                tools="tools" in capabilities,
                vision="vision" in capabilities,
                reasoning="thinking" in capabilities,
                programming_score=programming,
                structured_output=(previous.structured_output if previous else False),
                speed_tokens_second=(previous.speed_tokens_second if previous else None),
                ram_required_bytes=int(size * 1.18),
                vram_required_bytes=int(size * 1.10),
                verified=(previous.verified if previous else {}),
                failures=(previous.failures[-20:] if previous else []),
            )
            refreshed[name] = record
        self.records = refreshed
        self.save()
        return list(self.records.values())

    def add_remote_models(self, configurations: list[dict[str, Any]]) -> None:
        for config in configurations:
            name = str(config.get("name") or "").strip()
            if not name:
                continue
            capabilities = [str(item) for item in config.get("capabilities", ["completion"])]
            modalities = [str(item) for item in config.get("modalities", ["text"])]
            previous = self.records.get(name) or getattr(self, "_prior_records", {}).get(name)
            self.records[name] = ModelRecord(
                name=name,
                backend="openai_compatible",
                local=False,
                parameter_billions=float(config.get("parameter_billions") or 0),
                context_max=int(config.get("context_max") or 8192),
                capabilities=capabilities,
                modalities=modalities,
                tools="tools" in capabilities,
                vision="image" in modalities or "vision" in capabilities,
                reasoning="reasoning" in capabilities,
                programming_score=float(config.get("programming_score") or 0.6),
                structured_output="structured_output" in capabilities,
                languages=list(config.get("languages") or ["multilingual"]),
                speed_tokens_second=(previous.speed_tokens_second if previous else None),
                backend_required="OpenAI-compatible API",
                remote_model_id=str(config.get("model") or name),
                cost_input_per_million=float(config.get("cost_input_per_million") or 0),
                cost_output_per_million=float(config.get("cost_output_per_million") or 0),
                verified=(previous.verified if previous else {}),
                failures=(previous.failures[-20:] if previous else []),
            )
        self.save()

    def get(self, name: str) -> ModelRecord | None:
        return self.records.get(name)

    def all(self) -> list[ModelRecord]:
        return list(self.records.values())

    def record_run(self, name: str, output_tokens: int, elapsed_seconds: float, checks: dict[str, bool] | None = None) -> None:
        record = self.get(name)
        if not record:
            return
        if output_tokens > 0 and elapsed_seconds > 0:
            speed = output_tokens / elapsed_seconds
            if record.speed_tokens_second is None or record.speed_tokens_second < 0.2:
                record.speed_tokens_second = speed
            else:
                record.speed_tokens_second = record.speed_tokens_second * 0.7 + speed * 0.3
        for key, value in (checks or {}).items():
            if value:
                record.verified[key] = True
                if key == "structured_output":
                    record.structured_output = True
        self.save()

    def record_failure(self, name: str, category: str, detail: str, context_size: int | None = None) -> None:
        record = self.get(name)
        if not record:
            return
        record.failures.append({"category": category, "detail": detail[:300], "context_size": context_size})
        record.failures = record.failures[-20:]
        self.save()
