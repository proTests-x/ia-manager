from pathlib import Path

from pc_agent.learning import ProcedureLearner


def test_repeated_successful_sequence_becomes_safe_workflow(tmp_path: Path):
    learner = ProcedureLearner(tmp_path, threshold=3)
    for filename in ("a.txt", "b.txt", "c.txt"):
        learner.start_task()
        learner.observe("list_files", {"pattern": "**/*"}, True)
        learner.observe("read_file", {"path": filename}, True)
        learner.finish_task(True)

    tools = learner.list_tools()
    assert len(tools) == 1
    assert tools[0]["verified"] is True
    assert "step2_path" in tools[0]["parameters"]
    assert tools[0]["permissions"] == "heredadas por cada paso"


def test_failed_sequences_are_not_learned(tmp_path: Path):
    learner = ProcedureLearner(tmp_path, threshold=2)
    for _ in range(3):
        learner.start_task()
        learner.observe("run_terminal", {"command": "false"}, False)
        learner.observe("read_file", {"path": "x"}, False)
        learner.finish_task(False)
    assert learner.list_tools() == []

