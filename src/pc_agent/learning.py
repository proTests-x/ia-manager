from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProcedureLearner:
    """Observa secuencias; registra candidatos y herramientas verificadas fuera del núcleo."""

    def __init__(self, data_dir: Path, threshold: int = 3, enabled: bool = True):
        self.enabled = enabled
        self.threshold = max(2, threshold)
        self.history_path = data_dir / "action_history.jsonl"
        self.tools_dir = data_dir / "learned_tools"
        self.registry_path = self.tools_dir / "tools_registry.json"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.current: list[dict[str, Any]] = []
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"tools": [], "patterns": {}}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"tools": [], "patterns": {}}

    def _save(self) -> None:
        self.registry_path.write_text(json.dumps(self.registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def start_task(self) -> None:
        self.current = []

    def observe(self, name: str, arguments: dict[str, Any], ok: bool) -> None:
        if self.enabled:
            self.current.append({"name": name, "arguments": arguments, "ok": ok})

    def finish_task(self, success: bool) -> str | None:
        if not self.enabled or not self.current:
            return None
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "actions": self.current,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if not success or len(self.current) < 2:
            return None
        meta = self.registry.setdefault("meta", {})
        meta["completed_tasks"] = int(meta.get("completed_tasks", 0)) + 1
        signature = " -> ".join(action["name"] for action in self.current)
        patterns = self.registry.setdefault("patterns", {})
        item = patterns.setdefault(signature, {"count": 0, "successes": 0, "candidate": False})
        item["count"] += 1
        item["successes"] += 1
        samples = item.setdefault("samples", [])
        samples.append(self.current)
        item["samples"] = samples[-self.threshold:]
        if item["successes"] >= self.threshold:
            item["candidate"] = True
            self._auto_register(signature, item)
        if meta["completed_tasks"] % 25 == 0:
            self.consolidate()
        self._save()
        return signature if item.get("candidate") else None

    def _auto_register(self, signature: str, pattern: dict[str, Any]) -> None:
        samples = pattern.get("samples") or []
        if len(samples) < self.threshold:
            return
        if any("learned" in action.get("name", "") for action in samples[-1]):
            return
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
        name = "auto_" + "_".join(action["name"] for action in samples[-1][:3])[:35] + "_" + digest
        existing = next((tool for tool in self.list_tools() if tool.get("name") == name), None)
        parameters: list[str] = []
        steps: list[dict[str, Any]] = []
        for index, action in enumerate(samples[-1]):
            args: dict[str, Any] = {}
            source_args = action.get("arguments") or {}
            for key, value in source_args.items():
                observed = [(sample[index].get("arguments") or {}).get(key) for sample in samples if len(sample) > index]
                if len(observed) != len(samples) or any(isinstance(item, (dict, list)) for item in observed):
                    args[key] = value
                elif all(item == observed[0] for item in observed):
                    args[key] = observed[0]
                else:
                    parameter = f"step{index + 1}_{re.sub(r'[^a-zA-Z0-9_]', '_', key)}"
                    parameters.append(parameter)
                    args[key] = "{{" + parameter + "}}"
            steps.append({"tool": action["name"], "arguments": args})
        manifest = {
            "name": name,
            "description": f"Workflow aprendido automáticamente de: {signature}",
            "parameters": sorted(set(parameters)),
            "steps": steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uses": 0,
            "successes": 0,
            "failures": 0,
            "last_used": None,
            "success_rate": 1.0,
            "average_execution_seconds": 0.0,
            "average_time_saved_seconds": 0.0,
            "recent_errors": [],
            "observed_successes": int(pattern.get("successes", 0)),
            "verified": True,
            "permissions": "heredadas por cada paso",
            "enabled": True,
            "version": 1,
        }
        if existing:
            if existing.get("parameters") != manifest["parameters"] or existing.get("steps") != manifest["steps"]:
                existing["parameters"] = manifest["parameters"]
                existing["steps"] = manifest["steps"]
                existing["version"] = int(existing.get("version", 1)) + 1
                existing["verified"] = True
                existing["observed_successes"] = int(pattern.get("successes", 0))
                self.register(existing)
            return
        self.register(manifest)

    def consolidate(self) -> None:
        """Desactiva workflows defectuosos y fusiona duplicados estructurales."""
        unique: dict[str, dict[str, Any]] = {}
        for tool in self.list_tools():
            if int(tool.get("failures", 0)) >= 3:
                tool["enabled"] = False
            signature = json.dumps(tool.get("steps", []), ensure_ascii=False, sort_keys=True)
            previous = unique.get(signature)
            if previous is None or int(tool.get("successes", 0)) > int(previous.get("successes", 0)):
                unique[signature] = tool
        self.registry["tools"] = list(unique.values())

    def candidates_text(self) -> str:
        candidates = [name for name, data in self.registry.get("patterns", {}).items() if data.get("candidate")]
        if not candidates:
            return ""
        return "Patrones repetidos candidatos a automatización:\n- " + "\n- ".join(candidates[:10])

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.registry.get("tools", []))

    def register(self, manifest: dict[str, Any]) -> None:
        tools = self.registry.setdefault("tools", [])
        tools[:] = [tool for tool in tools if tool.get("name") != manifest.get("name")]
        tools.append(manifest)
        self._save()

    def unregister(self, name: str) -> None:
        tools = self.registry.setdefault("tools", [])
        tools[:] = [tool for tool in tools if tool.get("name") != name]
        self._save()

    def find(self, name: str) -> dict[str, Any] | None:
        for tool in self.list_tools():
            if tool.get("name") == name and tool.get("enabled", True):
                return tool
        return None
