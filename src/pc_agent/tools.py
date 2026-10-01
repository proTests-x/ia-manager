from __future__ import annotations

import ast
import ipaddress
import json
import mimetypes
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .learning import ProcedureLearner
from .policy import PermissionPolicy


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _schema("web_search", "Busca información actual en Internet.", {
        "query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    _schema("web_fetch", "Lee una URL pública y extrae texto. Úsala para verificar resultados de búsqueda.", {
        "url": {"type": "string"}}, ["url"]),
    _schema("run_terminal", "Ejecuta PowerShell en el workspace. La política central decide si es automático o requiere permiso.", {
        "command": {"type": "string"}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300}}, ["command"]),
    _schema("list_files", "Lista archivos del workspace con glob.", {
        "pattern": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 500}}, []),
    _schema("search_text", "Busca texto o una expresión regular dentro de archivos del workspace.", {
        "query": {"type": "string"}, "pattern": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 500}}, ["query"]),
    _schema("read_file", "Lee un archivo de texto dentro del workspace.", {
        "path": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 100, "maximum": 100000}}, ["path"]),
    _schema("write_file", "Crea o reemplaza un archivo de texto dentro del workspace.", {
        "path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _schema("create_directory", "Crea una carpeta dentro del workspace.", {
        "path": {"type": "string"}}, ["path"]),
    _schema("move_file", "Mueve o renombra un archivo o carpeta dentro del workspace sin sobrescribir.", {
        "source": {"type": "string"}, "destination": {"type": "string"}}, ["source", "destination"]),
    _schema("list_learned_workflows", "Busca workflows reutilizables aprendidos localmente.", {
        "query": {"type": "string"}}, []),
    _schema("register_learned_workflow", "Registra un workflow parametrizable compuesto sólo por herramientas existentes. Usa {{parametro}} en argumentos.", {
        "name": {"type": "string"}, "description": {"type": "string"},
        "parameters": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "object"}},
        "verification_arguments": {"type": "object"}},
        ["name", "description", "parameters", "steps", "verification_arguments"]),
    _schema("run_learned_workflow", "Ejecuta un workflow aprendido; cada paso conserva su política de permisos.", {
        "name": {"type": "string"}, "arguments": {"type": "object"}}, ["name", "arguments"]),
]


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str
    category: str = "ok"

    def as_text(self) -> str:
        return json.dumps({"ok": self.ok, "category": self.category, "output": self.output}, ensure_ascii=False)


class ToolRegistry:
    def __init__(
        self,
        settings: Settings,
        policy: PermissionPolicy,
        learner: ProcedureLearner | None = None,
        observer: Callable[[str, dict[str, Any], ToolResult], None] | None = None,
    ):
        self.settings = settings
        self.workspace = settings.workspace_path
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.learner = learner
        self.observer = observer or (lambda _name, _args, _result: None)
        self.http = httpx.Client(
            timeout=settings.web_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OllamaPCAgent/0.2)"},
        )

    @property
    def schemas(self) -> list[dict[str, Any]]:
        if self.learner and self.learner.enabled:
            return TOOL_SCHEMAS
        return [schema for schema in TOOL_SCHEMAS if "learned" not in schema["function"]["name"]]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            result = ToolResult(False, f"Herramienta desconocida: {name}", "unsupported_tool")
            self.observer(name, arguments, result)
            return result
        try:
            value = method(**arguments)
            result = value if isinstance(value, ToolResult) else ToolResult(True, str(value))
        except PermissionError as exc:
            result = ToolResult(False, f"Acción denegada: {exc}", "permission")
        except subprocess.TimeoutExpired as exc:
            result = ToolResult(False, f"Tiempo agotado después de {exc.timeout}s", "timeout")
        except Exception as exc:
            result = ToolResult(False, f"{type(exc).__name__}: {exc}", "tool_error")
        self.observer(name, arguments, result)
        return result

    def _safe_path(self, raw_path: str) -> Path:
        path = (self.workspace / raw_path).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise PermissionError("la ruta está fuera del workspace")
        return path

    def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("se requiere una URL http(s) válida")
        if self.settings.allow_private_web:
            return
        try:
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise ValueError("no se pudo resolver el dominio") from exc
        if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
            raise PermissionError("no se permiten direcciones privadas o locales")

    def tool_web_search(self, query: str, max_results: int = 5) -> str:
        self.policy.authorize("web_search", query)
        response = self.http.get("https://html.duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result"):
            anchor = item.select_one(".result__a")
            if not anchor:
                continue
            href = anchor.get("href", "")
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            snippet = item.select_one(".result__snippet")
            results.append({"title": anchor.get_text(" ", strip=True), "url": href,
                            "snippet": snippet.get_text(" ", strip=True) if snippet else ""})
            if len(results) >= max(1, min(int(max_results), 10)):
                break
        return json.dumps(results, ensure_ascii=False, indent=2) if results else "No se encontraron resultados."

    def tool_web_fetch(self, url: str) -> str:
        self.policy.authorize("web_fetch", url)
        current_url = url
        response = None
        for _ in range(6):
            self._validate_public_url(current_url)
            response = self.http.get(current_url)
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("location")
            if not location:
                break
            current_url = urljoin(str(response.url), location)
        else:
            raise ValueError("demasiadas redirecciones")
        assert response is not None
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type or not content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for node in soup(["script", "style", "noscript", "svg"]):
                node.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
            clean_text = re.sub(r"\n{3,}", "\n\n", text)
            result = f"Título: {title}\nURL final: {response.url}\n\n{clean_text}"
        elif content_type.startswith("text/") or "json" in content_type:
            result = response.text
        else:
            raise ValueError(f"tipo de contenido no legible: {content_type}")
        if len(result) > self.settings.max_page_chars:
            result = result[:self.settings.max_page_chars] + "\n[Contenido truncado]"
        return result

    def tool_run_terminal(self, command: str, timeout_seconds: int | None = None) -> ToolResult:
        if not command.strip():
            raise ValueError("el comando está vacío")
        self.policy.authorize("run_terminal", command)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(int(timeout_seconds or self.settings.terminal_timeout_seconds), 300),
        )
        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + "STDERR:\n" + completed.stderr
        output = output[:30_000] + ("\n[Salida truncada]" if len(output) > 30_000 else "")
        return ToolResult(completed.returncode == 0, f"exit_code={completed.returncode}\n{output or '[sin salida]'}",
                          "ok" if completed.returncode == 0 else "command_failed")

    def tool_list_files(self, pattern: str = "**/*", limit: int = 200) -> str:
        self.policy.authorize("list_files", pattern)
        files = []
        for path in self.workspace.glob(pattern):
            if path.is_file() and ".agent-data" not in path.parts and ".git" not in path.parts:
                files.append(str(path.relative_to(self.workspace)))
                if len(files) >= max(1, min(int(limit), 500)):
                    break
        return "\n".join(files) if files else "[sin archivos]"

    def tool_search_text(self, query: str, pattern: str = "**/*", limit: int = 200) -> str:
        self.policy.authorize("search_text", query)
        expression = re.compile(query)
        matches = []
        for path in self.workspace.glob(pattern):
            if not path.is_file() or ".git" in path.parts or ".agent-data" in path.parts or path.stat().st_size > 2_000_000:
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if expression.search(line):
                        matches.append(f"{path.relative_to(self.workspace)}:{number}: {line[:500]}")
                        if len(matches) >= max(1, min(int(limit), 500)):
                            return "\n".join(matches)
            except OSError:
                continue
        return "\n".join(matches) if matches else "[sin coincidencias]"

    def tool_read_file(self, path: str, max_chars: int = 30_000) -> str:
        self.policy.authorize("read_file", path)
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        if target.stat().st_size > 5_000_000:
            raise ValueError("el archivo supera 5 MB")
        guessed, _ = mimetypes.guess_type(target.name)
        if guessed and not (guessed.startswith("text/") or "json" in guessed):
            raise ValueError(f"parece un archivo binario ({guessed})")
        return target.read_text(encoding="utf-8", errors="replace")[:max(100, min(int(max_chars), 100_000))]

    def tool_write_file(self, path: str, content: str) -> str | ToolResult:
        self.policy.authorize("write_file", f"{path} ({len(content)} caracteres)")
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if target.suffix.lower() == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                return ToolResult(False, f"Archivo escrito, pero Python tiene sintaxis inválida: {exc}", "code_invalid")
        if target.suffix.lower() == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return ToolResult(False, f"Archivo escrito, pero JSON es inválido: {exc}", "structured_output")
        return f"Archivo escrito: {target.relative_to(self.workspace)}"

    def tool_create_directory(self, path: str) -> str:
        self.policy.authorize("create_directory", path)
        target = self._safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"Carpeta disponible: {target.relative_to(self.workspace)}"

    def tool_move_file(self, source: str, destination: str) -> str:
        self.policy.authorize("move_file", f"{source} -> {destination}")
        src, dst = self._safe_path(source), self._safe_path(destination)
        if not src.exists():
            raise FileNotFoundError(source)
        if dst.exists():
            raise FileExistsError(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Movido: {source} -> {destination}"

    def tool_list_learned_workflows(self, query: str = "") -> str:
        if not self.learner:
            return "Aprendizaje procedural deshabilitado."
        values = [tool for tool in self.learner.list_tools() if query.lower() in json.dumps(tool, ensure_ascii=False).lower()]
        return json.dumps(values, ensure_ascii=False, indent=2)

    def tool_register_learned_workflow(
        self, name: str, description: str, parameters: list[str],
        steps: list[dict[str, Any]], verification_arguments: dict[str, Any],
    ) -> str | ToolResult:
        if not self.learner:
            raise ValueError("aprendizaje procedural deshabilitado")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,50}", name):
            raise ValueError("nombre inválido")
        allowed = {schema["function"]["name"] for schema in self.schemas} - {
            "register_learned_workflow", "run_learned_workflow", "list_learned_workflows"}
        for step in steps:
            if step.get("tool") not in allowed or not isinstance(step.get("arguments", {}), dict):
                raise ValueError("el workflow contiene un paso no permitido")
        manifest = {"name": name, "description": description, "parameters": parameters, "steps": steps,
                    "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                    "last_used": None, "uses": 0, "successes": 0, "failures": 0,
                    "success_rate": 0.0, "average_execution_seconds": 0.0,
                    "average_time_saved_seconds": 0.0, "recent_errors": [],
                    "permissions": "heredadas por cada paso", "verified": False,
                    "enabled": True, "version": 1}
        self.learner.register(manifest)
        verification = self.tool_run_learned_workflow(name, verification_arguments)
        if not verification.ok:
            self.learner.unregister(name)
            return ToolResult(False, f"La verificación falló y el workflow no se registró: {verification.output}",
                              "workflow_verification_failed")
        verified = self.learner.find(name)
        assert verified is not None
        verified["verified"] = True
        self.learner.register(verified)
        return f"Workflow verificado mediante ejecución satisfactoria y registrado: {name}"

    def tool_run_learned_workflow(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self.learner:
            raise ValueError("aprendizaje procedural deshabilitado")
        manifest = self.learner.find(name)
        if not manifest:
            raise ValueError(f"workflow no encontrado: {name}")
        missing = [key for key in manifest.get("parameters", []) if key not in arguments]
        if missing:
            raise ValueError("faltan parámetros: " + ", ".join(missing))

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                for key, replacement in arguments.items():
                    value = value.replace("{{" + key + "}}", str(replacement))
                return value
            if isinstance(value, dict):
                return {key: substitute(item) for key, item in value.items()}
            if isinstance(value, list):
                return [substitute(item) for item in value]
            return value

        outputs = []
        started = time.perf_counter()
        manifest["uses"] = int(manifest.get("uses", 0)) + 1
        manifest["last_used"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        for step in manifest.get("steps", []):
            result = self.execute(str(step["tool"]), substitute(step.get("arguments", {})))
            outputs.append(f"{step['tool']}: {result.output}")
            if not result.ok:
                manifest["failures"] = int(manifest.get("failures", 0)) + 1
                errors = list(manifest.get("recent_errors", []))
                errors.append(result.output[:500])
                manifest["recent_errors"] = errors[-5:]
                manifest["success_rate"] = int(manifest.get("successes", 0)) / max(1, manifest["uses"])
                if manifest["failures"] >= 3:
                    manifest["enabled"] = False
                self.learner.register(manifest)
                return ToolResult(False, "\n".join(outputs), "workflow_failed")
        manifest["successes"] = int(manifest.get("successes", 0)) + 1
        elapsed = time.perf_counter() - started
        old_average = float(manifest.get("average_execution_seconds", 0.0))
        manifest["average_execution_seconds"] = elapsed if manifest["uses"] == 1 else old_average * 0.7 + elapsed * 0.3
        manifest["success_rate"] = manifest["successes"] / max(1, manifest["uses"])
        self.learner.register(manifest)
        return ToolResult(True, "\n".join(outputs))

    def close(self) -> None:
        self.http.close()
