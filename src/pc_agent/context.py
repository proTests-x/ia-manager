from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import ModelRecord
from .resources import ResourceSnapshot


IMPORTANT_RE = re.compile(
    r"(?i)(instrucci[oó]n|debes|no debes|decidimos|requisito|pendiente|error|fall[oó]|"
    r"traceback|todo|contrato|configuraci[oó]n|ruta|archivo|funci[oó]n|class\s|def\s|```|FIXME)"
)


def estimate_tokens(value: str | list[dict[str, Any]]) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = "\n".join(str(message.get("content") or "") for message in value)
    # Aproximación multilingüe conservadora sin depender de un tokenizer por arquitectura.
    return max(1, math.ceil(len(text) / 3.5))


@dataclass(slots=True)
class ContextBudget:
    model_max: int
    hardware_max: int
    effective_limit: int
    prompt_tokens: int
    response_reserved: int
    tool_tokens: int
    recovered_tokens: int = 0

    @property
    def percent(self) -> float:
        return min(100.0, self.prompt_tokens * 100 / max(self.effective_limit, 1))

    def display(self) -> str:
        return (
            f"Contexto: {self.prompt_tokens:,} / {self.effective_limit:,} ({self.percent:.0f} %) | "
            f"máx. modelo {self.model_max:,} | respuesta {self.response_reserved:,} | "
            f"herramientas {self.tool_tokens:,} | recuperados {self.recovered_tokens:,}"
        )


class ContextManager:
    def __init__(self, settings: Settings, resources: ResourceSnapshot):
        self.settings = settings
        self.resources = resources
        self.last_budget: ContextBudget | None = None

    def output_reserve(self, request: str) -> int:
        lower = request.lower()
        reserve = self.settings.response_reserve_min
        if any(word in lower for word in ("analiza", "razona", "compara", "diagnostica", "explica")):
            reserve = max(reserve, 2048)
        if any(word in lower for word in ("código", "codigo", "proyecto", "implementa", "archivo", "detallado")):
            reserve = max(reserve, 4096)
        if any(word in lower for word in ("completo", "exhaustivo", "migración", "refactor", "muchos archivos")) or len(request) > 5000:
            reserve = max(reserve, 8192)
        if any(word in lower for word in ("breve", "una frase", "resumen corto")):
            reserve = min(reserve, 1024)
        return min(reserve, self.settings.response_reserve_max)

    def hardware_context_max(self, model: ModelRecord) -> int:
        if not model.kv_bytes_per_token:
            return model.context_max
        # RAM y VRAM pueden participar juntas con offload de Ollama, pero se conserva margen
        # para SO, buffers de generación y picos de ejecución.
        current_free = int(self.resources.ram_available * 0.75 + self.resources.vram_free * 0.85)
        combined_total = self.resources.ram_total + self.resources.vram_total
        capacity_after_model = int(combined_total * 0.78 - model.size_bytes * 1.05 - 2 * 1024**3)
        # Si el modelo ya está cargado, la memoria libre es utilizable para ampliar KV;
        # si otro modelo ocupa memoria, la capacidad física estima lo recuperable al descargarlo.
        remaining = max(0, current_free, capacity_after_model)
        kv_limit = remaining // model.kv_bytes_per_token if remaining else self.settings.min_context_size
        return max(self.settings.min_context_size, min(model.context_max, int(kv_limit)))

    def prepare(
        self,
        messages: list[dict[str, Any]],
        model: ModelRecord,
        request: str,
        hard_cap: int | None = None,
    ) -> tuple[list[dict[str, Any]], ContextBudget]:
        reserve = self.output_reserve(request)
        override = self.settings.model_options(model.name).get("num_ctx")
        model_max = min(model.context_max, int(override)) if override else model.context_max
        if self.settings.context_size:
            model_max = min(model_max, int(self.settings.context_size))
        hardware_max = self.hardware_context_max(model)
        upper = min(model_max, hardware_max, hard_cap or model_max)
        failed_oom_contexts = [
            int(item["context_size"]) for item in model.failures
            if item.get("category") == "oom" and item.get("context_size")
        ]
        if failed_oom_contexts:
            upper = min(upper, max(self.settings.min_context_size, min(failed_oom_contexts) // 2))
        if self.settings.profile == "fast":
            upper = min(upper, 32768)
        elif self.settings.profile == "low_vram":
            upper = min(upper, 16384)
        raw_tokens = estimate_tokens(messages)
        needed = raw_tokens + reserve + max(512, raw_tokens // 10)
        if self.settings.auto_context:
            effective = max(self.settings.min_context_size, 2 ** math.ceil(math.log2(max(needed, 2))))
            effective = min(effective, upper)
        else:
            effective = upper
        effective = max(2048, effective)
        prepared, recovered = self._compress(messages, max(512, effective - reserve))
        prompt_tokens = estimate_tokens(prepared)
        # La reserva solicitada nunca puede desplazar por completo al prompt.
        reserve = min(reserve, max(256, effective - prompt_tokens))
        tool_tokens = sum(
            estimate_tokens(str(message.get("content") or ""))
            for message in prepared
            if message.get("role") == "tool"
        )
        budget = ContextBudget(model_max, hardware_max, effective, prompt_tokens, reserve, tool_tokens, recovered)
        self.last_budget = budget
        return prepared, budget

    def _compress(self, messages: list[dict[str, Any]], token_limit: int) -> tuple[list[dict[str, Any]], int]:
        before = estimate_tokens(messages)
        if before <= token_limit:
            return messages, 0

        system = [dict(message) for message in messages if message.get("role") == "system"]
        non_system = [dict(message) for message in messages if message.get("role") != "system"]
        recent = non_system[-10:]
        old = non_system[:-10]
        important: list[dict[str, Any]] = []
        summary_lines: list[str] = []
        seen: set[str] = set()
        for message in old:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            normalized = re.sub(r"\s+", " ", content)[:500]
            fingerprint = normalized.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            if message.get("role") == "tool":
                if len(summary_lines) < 20:
                    summary_lines.append(f"- resultado de herramienta: {normalized}")
            elif IMPORTANT_RE.search(content) and len(important) < 8:
                copy = dict(message)
                copy["content"] = content[:3000]
                important.append(copy)
            elif len(summary_lines) < 20:
                label = str(message.get("role") or "dato")
                summary_lines.append(f"- {label}: {normalized}")
        summary: list[dict[str, Any]] = []
        if summary_lines:
            summary.append({"role": "system", "content": "Resumen de contexto antiguo no redundante:\n" + "\n".join(summary_lines)})
        result = [*system, *summary, *important, *recent]
        while estimate_tokens(result) > token_limit and len(result) > len(system) + 3:
            # Quita primero resúmenes/historial menos reciente; conserva sistema y turno actual.
            del result[len(system)]
        if estimate_tokens(result) > token_limit:
            for message in result:
                if message.get("role") == "tool" and len(str(message.get("content") or "")) > 2000:
                    message["content"] = str(message["content"])[:2000] + "\n[resultado comprimido]"
        if estimate_tokens(result) > token_limit:
            for message in result[:-2]:
                if message.get("role") != "system" and len(str(message.get("content") or "")) > 1000:
                    message["content"] = str(message["content"])[:1000] + "\n[historial comprimido]"
        after = estimate_tokens(result)
        return result, max(0, before - after)
