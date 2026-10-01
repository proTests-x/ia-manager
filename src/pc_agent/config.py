from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PERMISSIONS = {
    "read_file": "AUTO",
    "list_files": "AUTO",
    "search_text": "AUTO",
    "write_file": "AUTO",
    "create_directory": "AUTO",
    "move_file": "NOTIFY",
    "web_search": "AUTO",
    "web_fetch": "AUTO",
    "terminal_safe": "AUTO",
    "terminal_project": "NOTIFY",
    "terminal_dangerous": "CONFIRM",
    "delete": "CONFIRM",
    "system_change": "CONFIRM",
    "remote_send": "CONFIRM",
}


@dataclass(slots=True)
class Settings:
    config_version: int = 2
    model: str = "srchmnmichael/qwen3.5-9B-uncensored:Q8_0"
    ollama_url: str = "http://127.0.0.1:11434"
    workspace: str = "."

    profile: str = "balanced"
    escalation_policy: str | None = None
    escalation_chain: list[str] = field(default_factory=list)
    max_model_attempts: int = 3
    show_router_messages: bool = True
    debug: bool = False

    auto_context: bool = True
    context_size: int | None = None
    min_context_size: int = 8192
    response_reserve_min: int = 1024
    response_reserve_max: int = 8192
    context_meter: bool = True
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    max_tool_rounds: int = 10
    web_timeout_seconds: int = 25
    terminal_timeout_seconds: int = 120
    max_page_chars: int = 30_000
    allow_private_web: bool = False
    permission_policy: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PERMISSIONS))

    remote_enabled: bool = False
    remote_allow_private_data: bool = False
    remote_confirm_before_send: bool = True
    remote_models: list[dict[str, Any]] = field(default_factory=list)

    learned_tools_enabled: bool = True
    pattern_threshold: int = 3
    system_prompt_extra: str = ""

    # Compatibilidad con la versión inicial.
    confirm_terminal: bool | None = None
    confirm_file_writes: bool | None = None

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def model_options(self, model: str) -> dict[str, Any]:
        override = self.model_overrides.get(model, {})
        return {
            "temperature": override.get("temperature", self.temperature),
            "top_p": override.get("top_p", self.top_p),
            "top_k": override.get("top_k", self.top_k),
            "num_ctx": override.get("context"),
            "num_predict": override.get("max_output_tokens"),
            "num_gpu": override.get("gpu_layers"),
            "num_thread": override.get("threads"),
            "num_batch": override.get("batch_size"),
            "reasoning_effort": override.get("reasoning_effort"),
            "backend_options": override.get("backend_options", {}),
        }

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            settings = cls()
            settings.save(path)
            return settings
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        known = cls.__dataclass_fields__
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"Opciones desconocidas en {path}: {', '.join(unknown)}")
        settings = cls(**raw)
        # En 0.1 context_size era siempre 32768. Al migrar, libera ese techo para que
        # el gestor automático use el máximo real del modelo y del hardware.
        if "auto_context" not in raw and raw.get("context_size") == 32768:
            settings.context_size = None
        if int(raw.get("config_version", 1)) < 2:
            # La política v2 materializa la autonomía solicitada; las banderas antiguas
            # quedan sólo para poder leer archivos previos.
            settings.permission_policy = dict(DEFAULT_PERMISSIONS)
            settings.confirm_file_writes = None
            settings.confirm_terminal = None
            settings.config_version = 2
        for key, value in DEFAULT_PERMISSIONS.items():
            settings.permission_policy.setdefault(key, value)
        settings.save(path)
        return settings
