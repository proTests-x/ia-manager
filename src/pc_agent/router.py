from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings
from .context import estimate_tokens
from .models import ModelRecord, ModelRegistry
from .resources import ResourceSnapshot
from .router_log import RouterLogger


@dataclass(slots=True)
class TaskRequirements:
    vision: bool = False
    image_generation: bool = False
    tools: bool = False
    web: bool = False
    programming: bool = False
    reasoning: bool = False
    structured_output: bool = False
    long_context: bool = False
    private: bool = False
    difficulty: int = 1
    estimated_tokens: int = 0


@dataclass(slots=True)
class RouteDecision:
    selected: str
    candidates: list[str]
    reason: str
    requirements: TaskRequirements
    rejected: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


PROFILE_RULES = {
    "fast": {"speed": 2.4, "quality": 0.6, "memory": 1.2, "max_params": 15},
    "balanced": {"speed": 1.0, "quality": 1.2, "memory": 0.8, "max_params": None},
    "max_quality": {"speed": 0.2, "quality": 2.5, "memory": 0.2, "max_params": None},
    "local_only": {"speed": 0.8, "quality": 1.3, "memory": 0.8, "local_only": True},
    "low_vram": {"speed": 1.4, "quality": 0.8, "memory": 2.5, "max_params": 15},
    "max_privacy": {"speed": 0.8, "quality": 1.0, "memory": 0.8, "local_only": True, "no_web": True},
}


class ModelRouter:
    def __init__(self, settings: Settings, registry: ModelRegistry, resources: ResourceSnapshot, logger: RouterLogger):
        self.settings = settings
        self.registry = registry
        self.resources = resources
        self.logger = logger

    def analyze(self, text: str, messages: list[dict[str, Any]], has_images: bool = False) -> TaskRequirements:
        lower = text.lower()
        programming = bool(re.search(r"\b(c[oó]digo|programa|python|javascript|typescript|bug|error|test|compila|refactor|api)\b", lower))
        reasoning = bool(re.search(r"\b(analiza|razona|demuestra|compara|arquitectura|planifica|diagn[oó]stico|complejo)\b", lower))
        web = bool(re.search(r"\b(actual|reciente|hoy|noticias|internet|web|documentaci[oó]n|versi[oó]n|precio|changelog|issue)\b|https?://", lower))
        if self.settings.profile == "max_privacy":
            web = False
        tools = web or programming or bool(re.search(r"\b(archivo|carpeta|proyecto|terminal|ejecuta|corrige|crea|modifica|instala)\b", lower))
        structured = bool(re.search(
            r"(en|como|formato|salida)\s+json\b|json\s+(válido|valido|estricto)|salida estructurada|json schema",
            lower,
        ))
        vision = has_images or bool(re.search(
            r"(analiza|describe|lee|mira).{0,40}(imagen|foto|captura|screenshot)|\.(png|jpe?g|webp)\b",
            lower,
        ))
        image_generation = bool(re.search(r"\b(genera|crea|dibuja)\b.{0,30}\b(imagen|foto|ilustraci[oó]n)\b", lower))
        private = bool(re.search(r"\b(secreto|contrase(?:ñ|n)a|token privado|confidencial|datos personales|api[_ -]?key)\b", lower))
        tokens = estimate_tokens(messages) + estimate_tokens(text)
        difficulty = 1 + int(programming) + int(reasoning) + int(len(text) > 1200) + int(tokens > 12000)
        return TaskRequirements(
            vision=vision,
            image_generation=image_generation,
            tools=tools,
            web=web,
            programming=programming,
            reasoning=reasoning,
            structured_output=structured,
            long_context=tokens > 24000,
            private=private,
            difficulty=min(difficulty, 5),
            estimated_tokens=tokens,
        )

    def route(
        self,
        text: str,
        messages: list[dict[str, Any]],
        pinned_model: str | None = None,
        has_images: bool = False,
    ) -> RouteDecision:
        req = self.analyze(text, messages, has_images)
        profile = PROFILE_RULES.get(self.settings.profile, PROFILE_RULES["balanced"])
        rejected: dict[str, str] = {}
        scores: dict[str, float] = {}
        eligible: list[tuple[float, ModelRecord]] = []
        for model in self.registry.all():
            reason = self._ineligible(model, req, profile)
            if reason:
                rejected[model.name] = reason
                continue
            quality = model.parameter_billions / 10
            if req.programming:
                quality += model.programming_score * 4
            if req.reasoning and model.reasoning:
                quality += 2
            if req.structured_output and (model.structured_output or model.tools):
                quality += 1
            speed = model.speed_tokens_second or (35 / max(model.parameter_billions, 3))
            memory_penalty = model.size_bytes / max(self.resources.ram_available + self.resources.vram_free, 1)
            api_cost = model.cost_input_per_million + model.cost_output_per_million
            score = (quality * profile.get("quality", 1) + speed * profile.get("speed", 1)
                     - memory_penalty * profile.get("memory", 1) - api_cost * 0.02)
            if not model.local:
                score -= 1.0  # Prefiere local si las capacidades son equivalentes.
            if req.difficulty <= 2:
                score -= max(0, model.parameter_billions - 12) * 0.12
            if model.name == self.settings.model:
                score += 0.15
            scores[model.name] = round(score, 3)
            eligible.append((score, model))
        eligible.sort(key=lambda item: item[0], reverse=True)
        ordered = [model.name for _, model in eligible]

        if self.settings.escalation_chain:
            chain = [name for name in self.settings.escalation_chain if name in ordered]
            ordered = chain + [name for name in ordered if name not in chain]
        if pinned_model:
            if pinned_model in ordered:
                ordered.remove(pinned_model)
                ordered.insert(0, pinned_model)
            elif pinned_model in self.registry.records:
                # Se conserva para que el agente pueda explicar la capacidad ausente y aplicar la política.
                ordered.insert(0, pinned_model)

        if not ordered:
            raise RuntimeError("Ningún modelo instalado satisface las capacidades y recursos requeridos.")
        selected = ordered[0]
        reason = self._reason(self.registry.get(selected), req, pinned_model)
        decision = RouteDecision(selected, ordered, reason, req, rejected, scores)
        self.logger.log("route", **asdict(decision))
        return decision

    def _ineligible(self, model: ModelRecord, req: TaskRequirements, profile: dict[str, Any]) -> str | None:
        if req.vision and not model.vision:
            return "no soporta visión"
        if req.image_generation and "image_generation" not in model.capabilities:
            return "no soporta generación de imágenes"
        if req.tools and not model.tools:
            return "no soporta function calling"
        if req.estimated_tokens > model.context_max * 0.85:
            return "contexto máximo insuficiente"
        if profile.get("local_only") and not model.local:
            return "el perfil exige ejecución local"
        if not model.local and not self.settings.remote_enabled:
            return "los modelos remotos están deshabilitados"
        if profile.get("max_params") and model.parameter_billions > profile["max_params"]:
            return "supera el tamaño permitido por el perfil"
        # Un modelo cargado puede estar consumiendo memoria ahora mismo; Ollama puede
        # descargarlo antes de cargar otro. Considera tanto memoria libre como capacidad
        # física razonablemente utilizable.
        available = max(
            self.resources.ram_available + self.resources.vram_free,
            int((self.resources.ram_total + self.resources.vram_total) * 0.90),
        )
        if model.ram_required_bytes > available * 1.25:
            return "memoria disponible insuficiente"
        if req.private and not model.local and not self.settings.remote_allow_private_data:
            return "la política de privacidad bloquea datos privados remotos"
        return None

    @staticmethod
    def _reason(model: ModelRecord | None, req: TaskRequirements, pinned: str | None) -> str:
        if not model:
            return "modelo disponible"
        if pinned == model.name:
            return "modelo fijado para la sesión"
        reasons = []
        if req.vision:
            reasons.append("soporta visión")
        if req.programming:
            reasons.append("adecuado para programación")
        if req.reasoning:
            reasons.append("soporta razonamiento")
        if req.tools:
            reasons.append("soporta herramientas")
        if req.long_context:
            reasons.append("contexto suficiente")
        return ", ".join(reasons) or "equilibrio entre capacidad, velocidad y memoria"
