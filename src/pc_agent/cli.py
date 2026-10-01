from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from .agent import Agent
from .config import Settings
from .context import ContextBudget, ContextManager
from .learning import ProcedureLearner
from .memory import MemoryStore
from .models import ModelRecord, ModelRegistry
from .ollama import OllamaClient, OllamaError
from .policy import PermissionPolicy
from .resources import ResourceDetector, ResourceSnapshot, human_bytes
from .router import PROFILE_RULES, ModelRouter
from .router_log import RouterLogger
from .tools import ToolRegistry, ToolResult


console = Console()


@dataclass(slots=True)
class Runtime:
    settings: Settings
    config_path: Path
    data_dir: Path
    memory: MemoryStore
    registry: ModelRegistry
    detector: ResourceDetector
    resources: ResourceSnapshot
    router: ModelRouter
    context: ContextManager
    logger: RouterLogger
    learner: ProcedureLearner
    agent: Agent


def configure_windows_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure:
                reconfigure(encoding="utf-8", errors="replace")


def approval(kind: str, detail: str) -> bool:
    console.print(Panel(detail, title=f"Aprobación requerida: {kind}", border_style="yellow"))
    try:
        return Confirm.ask("¿Permitir esta acción?", default=False, console=console)
    except EOFError:
        return False


def escalation_approval(reason: str, next_model: str) -> bool:
    console.print(Panel(reason, title="Escalado propuesto", border_style="yellow"))
    try:
        return Confirm.ask(f"¿Cambiar a {next_model}?", default=True, console=console)
    except EOFError:
        return False


def tool_notice(name: str, arguments: dict[str, Any]) -> None:
    safe_args = dict(arguments)
    if name == "write_file" and "content" in safe_args:
        safe_args["content"] = f"<{len(str(safe_args['content']))} caracteres>"
    console.print(f"[dim]-> {name} {json.dumps(safe_args, ensure_ascii=False)}[/dim]")


def router_notice(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def context_notice(settings: Settings, budget: ContextBudget) -> None:
    if settings.context_meter:
        console.print(f"[dim]{budget.display()}[/dim]")


HELP = """Comandos disponibles:
  /help                         Muestra esta ayuda
  /new                          Inicia una conversación nueva
  /model                        Muestra el modo actual
  /model auto                   Activa el router automático
  /model NOMBRE                 Fija un modelo para toda la sesión
  /model once NOMBRE            Fija un modelo sólo para la próxima tarea
  /models                       Muestra capacidades registradas
  /profile NOMBRE               fast, balanced, max_quality, local_only, low_vram, max_privacy
  /escalation MODO              never, ask o auto
  /context                      Muestra el último presupuesto de contexto
  /router                       Explica la última decisión del router
  /resources                    Vuelve a detectar RAM, VRAM, CPU y disco
  /image RUTA | PREGUNTA        Envía una imagen local a un modelo con visión
  /learned                      Lista workflows aprendidos
  /debug on|off                 Activa o desactiva logs detallados
  /remember TEXTO               Guarda un dato en memoria permanente
  /memories                     Lista recuerdos
  /forget ID                    Elimina un recuerdo
  /export [RUTA]                Exporta conversaciones para revisión
  /exit                          Cierra el agente
"""


def print_models(registry: ModelRegistry) -> None:
    for index, model in enumerate(registry.all(), 1):
        caps = []
        for name, enabled in (("tools", model.tools), ("vision", model.vision), ("reasoning", model.reasoning),
                              ("json-verificado", model.structured_output)):
            if enabled:
                caps.append(name)
        speed = f", {model.speed_tokens_second:.1f} tok/s" if model.speed_tokens_second else ""
        console.print(
            f"[cyan]{index}[/cyan]. {model.name}\n"
            f"   [dim]{model.parameter_billions:g}B {model.quantization}, contexto {model.context_max:,}, "
            f"{human_bytes(model.size_bytes)}{speed}; {', '.join(caps) or 'texto'}; "
            f"{'local' if model.local else 'remoto'} vía {model.backend_required}[/dim]"
        )


def select_start_model(registry: ModelRegistry, default_model: str) -> str | None:
    models = registry.all()
    if not models:
        raise OllamaError("Ollama no tiene modelos instalados. Usa: ollama pull NOMBRE")
    console.print("\n[bold cyan]Modo de modelo:[/bold cyan]")
    console.print("  [cyan]0[/cyan]. Automático [dim](router por capacidades, recomendado)[/dim]")
    default_index = 0
    for index, model in enumerate(models, 1):
        if model.name == default_model:
            default_index = index
        caps = ", ".join(model.modalities + (["tools"] if model.tools else []))
        console.print(f"  [cyan]{index}[/cyan]. {model.name} [dim]({human_bytes(model.size_bytes)}, {caps})[/dim]")
    while True:
        choice = console.input("\n[bold]Elige modo/modelo[/bold] [dim](Enter = automático)[/dim]: ").strip()
        if not choice or choice == "0":
            return None
        try:
            selected = int(choice)
        except ValueError:
            console.print("[yellow]Escribe un número.[/yellow]")
            continue
        if 1 <= selected <= len(models):
            return models[selected - 1].name
        console.print(f"[yellow]Elige un número entre 0 y {len(models)}.[/yellow]")


def configure_escalation(settings: Settings, config_path: Path, interactive: bool) -> None:
    if not interactive:
        if settings.escalation_policy not in {"never", "ask", "auto"}:
            settings.escalation_policy = "ask"
            settings.save(config_path)
        return
    current = settings.escalation_policy if settings.escalation_policy in {"never", "ask", "auto"} else "ask"
    default_choice = {"never": "1", "ask": "2", "auto": "3"}[current]
    console.print("\n[bold cyan]¿Cómo quieres usar el escalado dinámico en esta sesión?[/bold cyan]")
    console.print("  [cyan]1[/cyan]. Nunca escalar")
    console.print("  [cyan]2[/cyan]. Preguntarme antes de escalar [dim](recomendado)[/dim]")
    console.print("  [cyan]3[/cyan]. Escalar automáticamente")
    while True:
        choice = console.input(
            f"[bold]Elige una política[/bold] [dim](Enter = {default_choice}, guardada: {current})[/dim]: "
        ).strip() or default_choice
        values = {"1": "never", "2": "ask", "3": "auto"}
        if choice in values:
            settings.escalation_policy = values[choice]
            settings.save(config_path)
            return
        console.print("[yellow]Elige 1, 2 o 3.[/yellow]")


def find_model(value: str, registry: ModelRegistry) -> str | None:
    if value in registry.records:
        return value
    matches = [name for name in registry.records if value.lower() in name.lower()]
    return matches[0] if len(matches) == 1 else None


def handle_command(text: str, runtime: Runtime) -> bool:
    command, _, value = text.partition(" ")
    value = value.strip()
    agent, settings = runtime.agent, runtime.settings
    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        console.print(HELP)
    elif command == "/new":
        agent.new_session()
        console.print(f"[green]Nueva sesión:[/green] {agent.session_id}")
    elif command == "/model":
        if not value:
            console.print(f"Modelo: {agent.pinned_model or 'automático'}; actual: {agent.current_model or 'ninguno'}")
        elif value == "auto":
            agent.set_pinned_model(None)
            console.print("[green]Router automático activado.[/green]")
        elif value.startswith("once "):
            selected = find_model(value[5:].strip(), runtime.registry)
            if selected:
                agent.set_next_model(selected)
                console.print(f"[green]Próxima tarea fijada a {selected}.[/green]")
            else:
                console.print("[yellow]Modelo no encontrado o nombre ambiguo.[/yellow]")
        else:
            selected = find_model(value, runtime.registry)
            if selected:
                agent.set_pinned_model(selected)
                console.print(f"[green]Sesión fijada a {selected}.[/green]")
            else:
                console.print("[yellow]Modelo no encontrado o nombre ambiguo.[/yellow]")
    elif command == "/models":
        print_models(runtime.registry)
    elif command == "/profile":
        if not value:
            console.print(f"Perfil actual: [cyan]{settings.profile}[/cyan]. Disponibles: {', '.join(PROFILE_RULES)}")
        elif value in PROFILE_RULES:
            settings.profile = value
            settings.save(runtime.config_path)
            console.print(f"[green]Perfil guardado: {value}.[/green]")
        else:
            console.print(f"[yellow]Perfil desconocido. Usa: {', '.join(PROFILE_RULES)}[/yellow]")
    elif command == "/escalation":
        if value in {"never", "ask", "auto"}:
            settings.escalation_policy = value
            settings.save(runtime.config_path)
            console.print(f"[green]Escalado: {value}.[/green]")
        else:
            console.print(f"Escalado actual: {settings.escalation_policy}. Opciones: never, ask, auto")
    elif command == "/context":
        console.print(runtime.context.last_budget.display() if runtime.context.last_budget else "Aún no hay medición.")
    elif command == "/router":
        if agent.last_decision:
            console.print_json(json.dumps(asdict(agent.last_decision), ensure_ascii=False))
        else:
            console.print("Aún no hay una decisión del router.")
    elif command == "/resources":
        runtime.resources = runtime.detector.detect()
        runtime.router.resources = runtime.resources
        runtime.context.resources = runtime.resources
        show_resources(runtime.resources)
    elif command == "/learned":
        tools = runtime.learner.list_tools()
        console.print_json(json.dumps(tools, ensure_ascii=False)) if tools else console.print("[dim]No hay workflows aprendidos.[/dim]")
    elif command == "/debug":
        if value in {"on", "off"}:
            settings.debug = value == "on"
            runtime.logger.enabled = settings.debug
            settings.save(runtime.config_path)
            console.print(f"[green]Debug {value}.[/green]")
        else:
            console.print(f"Debug: {'on' if settings.debug else 'off'}")
    elif command == "/image":
        raw_path, separator, prompt = value.partition("|")
        if not separator or not raw_path.strip() or not prompt.strip():
            console.print("[yellow]Uso: /image RUTA | PREGUNTA[/yellow]")
        else:
            image_path = (settings.workspace_path / raw_path.strip()).resolve()
            if settings.workspace_path not in image_path.parents and image_path != settings.workspace_path:
                console.print("[red]La imagen debe estar dentro del workspace.[/red]")
            elif not image_path.is_file():
                console.print("[red]Imagen no encontrada.[/red]")
            else:
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                response = agent.ask(prompt.strip(), images=[encoded])
                console.print("\n[bold green]Agente>[/bold green]")
                console.print(Markdown(response))
    elif command == "/remember":
        if value:
            memory_id = runtime.memory.remember(value)
            console.print(f"[green]Recuerdo guardado con ID {memory_id}.[/green]")
        else:
            console.print("[yellow]Uso: /remember TEXTO[/yellow]")
    elif command == "/memories":
        records = runtime.memory.list_memories()
        for record in records:
            console.print(f"[cyan]{record['id']}[/cyan]  {record['text']}")
        if not records:
            console.print("[dim]No hay recuerdos.[/dim]")
    elif command == "/forget":
        try:
            removed = runtime.memory.forget(int(value))
            console.print("[green]Eliminado.[/green]" if removed else "[yellow]ID no encontrado.[/yellow]")
        except ValueError:
            console.print("[yellow]Uso: /forget ID_NUMÉRICO[/yellow]")
    elif command == "/export":
        target = Path(value).expanduser().resolve() if value else runtime.data_dir / "training_export.jsonl"
        count = runtime.memory.export_jsonl(target)
        console.print(f"[green]Exportadas {count} conversaciones a {target}[/green]")
    else:
        console.print("[yellow]Comando desconocido. Usa /help.[/yellow]")
    return True


def show_resources(resources: ResourceSnapshot) -> None:
    gpu = ", ".join(f"{item.name} ({human_bytes(item.free_bytes)} libres)" for item in resources.gpus) or "sin GPU NVIDIA detectada"
    console.print(
        f"[dim]CPU: {resources.cpu}, {resources.physical_cores} núcleos/{resources.logical_threads} hilos | "
        f"RAM: {human_bytes(resources.ram_available)} libres de {human_bytes(resources.ram_total)} | "
        f"GPU: {gpu} | Disco: {human_bytes(resources.disk_free)} libres[/dim]"
    )
    console.print(f"[dim]Backends detectados: {', '.join(resources.backends)}[/dim]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente local Ollama con router inteligente")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prompt", help="Ejecuta una sola petición y termina")
    parser.add_argument("--model", help="Fija un modelo para esta ejecución; usa 'auto' para router")
    parser.add_argument("--profile", choices=list(PROFILE_RULES))
    return parser


def main() -> int:
    configure_windows_console()
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        settings = Settings.load(config_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        console.print(f"[red]Configuración inválida:[/red] {exc}")
        return 2
    if args.profile:
        settings.profile = args.profile
    configure_escalation(settings, config_path, interactive=not bool(args.prompt))

    data_dir = config_path.parent / ".agent-data"
    memory = MemoryStore(data_dir / "memory.sqlite3")
    ollama = OllamaClient(settings.ollama_url)
    tools: ToolRegistry | None = None
    try:
        tags = ollama.health()
        detector = ResourceDetector(settings.workspace_path)
        resources = detector.detect()
        registry = ModelRegistry(data_dir / "model_registry.json")
        registry.refresh_ollama(ollama, tags)
        registry.add_remote_models(settings.remote_models)
        show_resources(resources)

        if args.model:
            pinned = None if args.model == "auto" else find_model(args.model, registry)
            if args.model != "auto" and not pinned:
                console.print("[red]Modelo indicado no encontrado o ambiguo.[/red]")
                return 2
        elif args.prompt:
            pinned = None
        else:
            pinned = select_start_model(registry, settings.model)

        logger = RouterLogger(data_dir / "router-debug.jsonl", settings.debug)
        logger.log("resources", **resources.to_dict())
        learner = ProcedureLearner(data_dir, settings.pattern_threshold, settings.learned_tools_enabled)
        policy = PermissionPolicy(settings, approval, lambda message: console.print(f"[dim]Permitido: {message}[/dim]"))

        def observe(name: str, arguments: dict[str, Any], result: ToolResult) -> None:
            learner.observe(name, arguments, result.ok)
            logger.log("tool", name=name, arguments=arguments, ok=result.ok, category=result.category)

        tools = ToolRegistry(settings, policy, learner, observe)
        router = ModelRouter(settings, registry, resources, logger)
        context = ContextManager(settings, resources)
        agent = Agent(
            settings, memory, ollama, tools, router, context, registry, logger, learner, detector,
            pinned_model=pinned, on_tool=tool_notice, on_router=router_notice,
            on_context=lambda budget: context_notice(settings, budget), confirm_escalation=escalation_approval,
        )
        runtime = Runtime(settings, config_path, data_dir, memory, registry, detector, resources,
                          router, context, logger, learner, agent)
        if args.prompt:
            console.print(Markdown(agent.ask(args.prompt)))
            return 0

        mode = pinned or "automático"
        console.print(Panel(
            f"Modo: {mode}\nPerfil: {settings.profile}\nEscalado: {settings.escalation_policy}\n"
            f"Workspace: {settings.workspace_path}\nSesión: {agent.session_id}\nEscribe /help para ver comandos.",
            title="Ollama PC Agent 0.2", border_style="cyan"))
        while True:
            try:
                text = console.input("\n[bold cyan]Tú>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nHasta luego.")
                break
            if not text:
                continue
            if text.startswith("/"):
                if not handle_command(text, runtime):
                    break
                continue
            try:
                response = agent.ask(text)
                console.print("\n[bold green]Agente>[/bold green]")
                console.print(Markdown(response))
            except OllamaError as exc:
                console.print(f"[red]{exc}[/red]")
            except Exception as exc:
                console.print(f"[red]Error inesperado: {type(exc).__name__}: {exc}[/red]")
        return 0
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    finally:
        if tools:
            tools.close()
        ollama.close()
        memory.close()


if __name__ == "__main__":
    sys.exit(main())
