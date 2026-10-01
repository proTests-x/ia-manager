from pathlib import Path

from pc_agent.config import Settings
from pc_agent.policy import PermissionPolicy
from pc_agent.tools import ToolRegistry


def make_tools(tmp_path: Path, confirm=lambda *_: True):
    settings = Settings(workspace=str(tmp_path))
    policy = PermissionPolicy(settings, confirm)
    return ToolRegistry(settings, policy)


def test_path_cannot_escape_workspace(tmp_path: Path):
    tools = make_tools(tmp_path)
    result = tools.execute("read_file", {"path": "../secret.txt"})
    assert not result.ok
    assert "fuera del workspace" in result.output
    tools.close()


def test_write_and_read_file(tmp_path: Path):
    tools = make_tools(tmp_path)
    written = tools.execute("write_file", {"path": "notes/a.txt", "content": "hola"})
    read = tools.execute("read_file", {"path": "notes/a.txt"})
    assert written.ok
    assert read.ok and read.output == "hola"
    tools.close()


def test_terminal_can_be_denied(tmp_path: Path):
    settings = Settings(workspace=str(tmp_path))
    settings.permission_policy["terminal_safe"] = "CONFIRM"
    tools = ToolRegistry(settings, PermissionPolicy(settings, lambda *_: False))
    result = tools.execute("run_terminal", {"command": "Write-Output no-debe-ejecutarse"})
    assert not result.ok
    assert "no aprobó" in result.output
    tools.close()


def test_python_write_is_syntax_checked(tmp_path: Path):
    tools = make_tools(tmp_path)
    result = tools.execute("write_file", {"path": "broken.py", "content": "def nope(:\n    pass"})
    assert not result.ok
    assert result.category == "code_invalid"
    tools.close()
