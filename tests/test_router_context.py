from pathlib import Path

from pc_agent.config import Settings
from pc_agent.context import ContextManager
from pc_agent.models import ModelRecord, ModelRegistry
from pc_agent.resources import ResourceSnapshot
from pc_agent.router import ModelRouter
from pc_agent.router_log import RouterLogger


def resources():
    return ResourceSnapshot(
        cpu="test", physical_cores=4, logical_threads=8,
        ram_total=64 * 1024**3, ram_available=48 * 1024**3,
        disk_free=100 * 1024**3,
    )


def test_router_requires_capability_not_parameter_count(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "models.json")
    registry.records = {
        "large-text": ModelRecord(name="large-text", parameter_billions=70, size_bytes=10, tools=True, vision=False),
        "small-vision": ModelRecord(name="small-vision", parameter_billions=9, size_bytes=10, tools=True, vision=True),
    }
    settings = Settings(profile="balanced")
    router = ModelRouter(settings, registry, resources(), RouterLogger(tmp_path / "log.jsonl", False))
    decision = router.route("Analiza esta imagen", [], has_images=True)
    assert decision.selected == "small-vision"
    assert decision.rejected["large-text"] == "no soporta visión"


def test_dynamic_context_respects_model_max_and_reserves_output():
    settings = Settings(min_context_size=2048, context_size=None)
    manager = ContextManager(settings, resources())
    model = ModelRecord(name="test", context_max=8192, size_bytes=1, kv_bytes_per_token=1)
    messages = [{"role": "user", "content": "hola" * 100}]
    _, budget = manager.prepare(messages, model, "Escribe código Python completo")
    assert budget.effective_limit <= 8192
    assert budget.response_reserved >= 4096
    assert budget.prompt_tokens + budget.response_reserved <= budget.effective_limit

