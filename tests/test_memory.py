from pathlib import Path

from pc_agent.memory import MemoryStore


def test_memory_retrieval(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    store.remember("Mi lenguaje favorito para automatización es Python")
    store.remember("Prefiero respuestas cortas")
    assert "Python" in store.relevant("¿Cuál es mi lenguaje favorito?")[0]
    store.close()


def test_session_export(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    session = store.new_session()
    store.add_message(session, "user", "hola")
    store.add_message(session, "assistant", "hola")
    target = tmp_path / "export.jsonl"
    assert store.export_jsonl(target) == 1
    assert '"messages"' in target.read_text(encoding="utf-8")
    store.close()

