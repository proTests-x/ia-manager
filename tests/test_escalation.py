from pathlib import Path

from pc_agent.agent import Agent
from pc_agent.config import Settings
from pc_agent.context import ContextManager
from pc_agent.learning import ProcedureLearner
from pc_agent.memory import MemoryStore
from pc_agent.models import ModelRecord, ModelRegistry
from pc_agent.ollama import ChatResponse
from pc_agent.policy import PermissionPolicy
from pc_agent.resources import ResourceDetector, ResourceSnapshot
from pc_agent.router import ModelRouter
from pc_agent.router_log import RouterLogger
from pc_agent.tools import ToolRegistry


class FakeOllama:
    def __init__(self):
        self.called = []

    def chat(self, **kwargs):
        self.called.append(kwargs["model"])
        return ChatResponse(
            message={"role": "assistant", "content": "imagen analizada"},
            prompt_tokens=100,
            output_tokens=5,
            elapsed_seconds=0.1,
            generation_seconds=0.1,
        )

    def unload(self, model):
        pass


def test_pinned_incapable_model_escalates_by_capability(tmp_path: Path):
    settings = Settings(workspace=str(tmp_path), escalation_policy="auto", context_meter=False)
    resources = ResourceSnapshot("test", 4, 8, 64 * 1024**3, 48 * 1024**3, 100 * 1024**3)
    registry = ModelRegistry(tmp_path / "models.json")
    registry.records = {
        "text-only": ModelRecord(name="text-only", tools=True, vision=False, context_max=8192),
        "vision-model": ModelRecord(name="vision-model", tools=True, vision=True, context_max=8192),
    }
    logger = RouterLogger(tmp_path / "router.jsonl", False)
    router = ModelRouter(settings, registry, resources, logger)
    context = ContextManager(settings, resources)
    learner = ProcedureLearner(tmp_path / "data")
    policy = PermissionPolicy(settings, lambda *_: True)
    tools = ToolRegistry(settings, policy, learner)
    memory = MemoryStore(tmp_path / "memory.db")
    fake = FakeOllama()
    notices = []
    agent = Agent(
        settings, memory, fake, tools, router, context, registry, logger, learner,
        ResourceDetector(tmp_path), pinned_model="text-only", on_router=notices.append,
    )

    answer = agent.ask("Analiza esta imagen", images=["AA=="])

    assert answer == "imagen analizada"
    assert fake.called == ["vision-model"]
    assert any("no soporta visión" in notice for notice in notices)
    tools.close()
    memory.close()

