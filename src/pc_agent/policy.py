from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import Settings


class PermissionLevel(str, Enum):
    AUTO = "AUTO"
    NOTIFY = "NOTIFY"
    CONFIRM = "CONFIRM"
    BLOCK = "BLOCK"


@dataclass(slots=True)
class PermissionDecision:
    level: PermissionLevel
    category: str
    reason: str


class PermissionPolicy:
    DANGEROUS = re.compile(
        r"(?i)(remove-item|\brm\b|\bdel\b|rmdir|format-volume|clear-disk|"
        r"git\s+(reset\s+--hard|clean\s+-[a-z]*f)|shutdown|restart-computer|"
        r"stop-computer|reg\s+(delete|add)|set-executionpolicy|diskpart)"
    )
    BLOCKED = re.compile(r"(?i)(invoke-expression|\biex\b).*(https?://)|format\s+[a-z]:")
    SYSTEM_CHANGE = re.compile(
        r"(?i)(winget\s+install|choco\s+install|pip\s+install(?!\s+-e\s+\.)|"
        r"npm\s+install\s+-g|new-localuser|set-itemproperty.*registry)"
    )
    SAFE = re.compile(
        r"(?i)^\s*(get-|select-|where-|measure-|compare-|test-|resolve-|"
        r"write-output|echo\b|pwd\b|ls\b|dir\b|tree\b|type\b|rg\b|git\s+(status|diff|log|show)\b|"
        r"pytest\b|ruff\b|mypy\b|npm\s+test\b|dotnet\s+test\b|"
        r"python\s+-m\s+(pytest|compileall|pip\s+check)\b)"
    )

    def __init__(
        self,
        settings: Settings,
        confirm: Callable[[str, str], bool],
        notify: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.confirm = confirm
        self.notify = notify or (lambda _message: None)

    def classify_terminal(self, command: str) -> PermissionDecision:
        if self.BLOCKED.search(command):
            return PermissionDecision(PermissionLevel.BLOCK, "terminal_dangerous", "patrón bloqueado por seguridad")
        if self.DANGEROUS.search(command):
            return self._decision("terminal_dangerous", "comando destructivo o crítico")
        if self.SYSTEM_CHANGE.search(command):
            return self._decision("system_change", "modifica paquetes o configuración del sistema")
        if self.SAFE.search(command):
            return self._decision("terminal_safe", "comando de inspección o verificación")
        return self._decision("terminal_project", "operación ordinaria dentro del proyecto")

    def decide(self, action: str, detail: str = "") -> PermissionDecision:
        if action in {"web_search", "web_fetch"} and self.settings.profile == "max_privacy":
            return PermissionDecision(PermissionLevel.BLOCK, action, "el perfil de privacidad máxima bloquea Internet")
        if action == "remote_send":
            if not self.settings.remote_enabled:
                return PermissionDecision(PermissionLevel.BLOCK, action, "los modelos remotos están deshabilitados")
            if not self.settings.remote_allow_private_data and re.search(
                r"(?i)(secreto|contraseña|token|api[_ -]?key|confidencial)", detail
            ):
                return PermissionDecision(PermissionLevel.BLOCK, action, "la política impide enviar datos privados")
            if self.settings.remote_confirm_before_send:
                return PermissionDecision(PermissionLevel.CONFIRM, action, "la política exige confirmar cada envío remoto")
        if action == "run_terminal":
            return self.classify_terminal(detail)
        return self._decision(action, f"política configurada para {action}")

    def authorize(self, action: str, detail: str = "") -> PermissionDecision:
        decision = self.decide(action, detail)
        if decision.level is PermissionLevel.BLOCK:
            raise PermissionError(decision.reason)
        if decision.level is PermissionLevel.CONFIRM and not self.confirm(decision.category, detail):
            raise PermissionError("el usuario no aprobó la acción")
        if decision.level is PermissionLevel.NOTIFY:
            self.notify(f"{decision.category}: {detail}")
        return decision

    def _decision(self, category: str, reason: str) -> PermissionDecision:
        raw = self.settings.permission_policy.get(category, "CONFIRM").upper()
        try:
            level = PermissionLevel(raw)
        except ValueError:
            level = PermissionLevel.CONFIRM
        return PermissionDecision(level, category, reason)
