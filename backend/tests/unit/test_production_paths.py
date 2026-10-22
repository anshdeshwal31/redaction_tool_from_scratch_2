"""Production never runs the benchmark adapter's code: ``mlredact run`` and the HTTP API (which starts ``mlredact run``
for every job, src/mlredact/api/runner.py) hand the runner no injected stage inputs; outside the benchmark's own
commands, the injected worker task and the runner's branch for injected inputs, no code imports the modules that
build them (``mlredact.bench``, ``mlredact.pipeline.injection``) or names the injected worker task, inside functions
too; and a production job neither imports nor runs them.

The production job runs as in tests/unit/test_bench_runner.py: its in-process worker pool and its configuration
without models, on its synthetic letter.  That pool stands in for the worker processes and for page analysis, so the
source check above covers what the job cannot show: worker start-up, OCR and the branches the letter never takes.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any

import pytest
import yaml
from test_bench_runner import LETTER, NO_MODELS, InProcessPool
from typer.testing import CliRunner

from mlredact.cli.main import app

SRC = Path(__file__).resolve().parents[2] / "src" / "mlredact"
# What production must never use: the benchmark adapter and the injected stage inputs.
ADAPTER_MODULES = ("mlredact.bench", "mlredact.pipeline.injection")
ADAPTER_FILES = (SRC / "bench", SRC / "pipeline" / "injection.py")
# The worker task that reads injected text instead of running OCR.
INJECTED_TASK = "task_analyse_injected"
# mlredact's commands, and the commands and options only the benchmark gives it.
COMMANDS = frozenset(
    {c.name or c.callback.__name__.replace("_", "-") for c in app.registered_commands if c.callback is not None}
    | {g.name for g in app.registered_groups if g.name}
)
BENCH_ONLY = frozenset({"bench", "bench-version", "--mode", "--inject-dir"})
# Where production code may use the adapter, as (module, function; "" for the module's own code): the benchmark's
# commands, the runner (its import of the injected worker task, and its branch for injected inputs) and that task.
ADAPTER_USERS = frozenset(
    {
        ("mlredact.cli.main", "bench"),
        ("mlredact.cli.main", "bench_version"),
        ("mlredact.pipeline.runner", ""),
        ("mlredact.pipeline.runner", "Redactor.run"),
        ("mlredact.pipeline.workers", INJECTED_TASK),
    }
)


def _adapter_module(name: str) -> bool:
    return any(name == m or name.startswith(f"{m}.") for m in ADAPTER_MODULES)


def _adapter_file(filename: str) -> bool:
    path = Path(filename).resolve()
    return any(path == f or f in path.parents for f in ADAPTER_FILES)


# --------------------------------------------------------------------------------------------- the code
def _name(node: ast.AST) -> str | None:
    """The name a node refers to: a variable, an attribute, an imported name or a string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.alias):
        return node.name
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _uses_of_the_injection(tree: ast.AST, package: str) -> list[str]:
    """What in ``tree`` (code of ``package``) could hand the runner injected stage inputs: an ``inject`` argument,
    keyword arguments unpacked into a ``run`` call, the injected worker task (by name, or as a string), an import of
    the adapter's modules (wherever it is, inside functions too)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "inject":
            found.append(f"inject= at line {node.lineno}")
        elif isinstance(node, ast.Call) and _name(node.func) == "run" and any(k.arg is None for k in node.keywords):
            found.append(f"** into run() at line {node.lineno}")
        elif _name(node) == INJECTED_TASK:
            found.append(f"{INJECTED_TASK} at line {getattr(node, 'lineno', '?')}")
        if isinstance(node, ast.ImportFrom):
            module = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            names = [module, *(f"{module}.{a.name}" for a in node.names)]
            found += [f"import of {n} at line {node.lineno}" for n in names if _adapter_module(n)]
        elif isinstance(node, ast.Import):
            found += [f"import of {a.name} at line {node.lineno}" for a in node.names if _adapter_module(a.name)]
    return found


def _command(name: str) -> ast.FunctionDef:
    """The function behind a command of mlredact's command line (src/mlredact/cli/main.py)."""
    tree = ast.parse((SRC / "cli" / "main.py").read_text("utf-8"))
    (found,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return found


def _api_modules() -> list[Path]:
    return sorted((SRC / "api").glob("*.py"))


def test_the_run_command_never_passes_injected_inputs() -> None:
    assert _uses_of_the_injection(_command("run"), "mlredact.cli") == []
    # The check finds them where they are: in the benchmark adapter's command.
    uses = _uses_of_the_injection(_command("bench"), "mlredact.cli")
    assert any(u.startswith("inject=") for u in uses)
    assert any(u.startswith("import of mlredact.bench.oracle.OracleInputs") for u in uses)


@pytest.mark.parametrize("path", _api_modules(), ids=lambda p: p.name)
def test_the_api_never_passes_injected_inputs(path: Path) -> None:
    assert _uses_of_the_injection(ast.parse(path.read_text("utf-8")), "mlredact.api") == []


def _adapter_uses(tree: ast.Module, package: str) -> set[str]:
    """Where in a module's code (function by qualified name, "" for the module's own code) one of the adapter's
    modules or the injected worker task is imported or named (as a variable, an attribute or a string), inside
    functions too; imports for type checking only are not counted."""
    found: set[str] = set()

    def visit(node: ast.AST, where: str) -> None:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            children: list[ast.AST] = list(node.orelse)
        else:
            children = list(ast.iter_child_nodes(node))
            names = [_name(node) or ""]
            if isinstance(node, ast.ImportFrom):
                module = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                names = [module, *(f"{module}.{a.name}" for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(_adapter_module(n) or n.rsplit(".", 1)[-1] == INJECTED_TASK for n in names):
                found.add(where)
        for child in children:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, f"{where}.{child.name}" if where else child.name)
            else:
                visit(child, where)

    visit(tree, "")
    return found


def test_production_code_uses_the_adapter_only_where_the_benchmark_runs_it() -> None:
    used = set()
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(path.relative_to(SRC.parent).with_suffix("").parts)
        if _adapter_module(module):
            continue
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        used |= {(module, where) for where in _adapter_uses(ast.parse(path.read_text("utf-8")), package)}
    # Exactly there: the check finds each of them.
    assert used == ADAPTER_USERS


def test_the_api_starts_the_run_command_and_never_the_benchmark_s() -> None:
    started = []
    for path in _api_modules():
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            assert not (isinstance(node, ast.Constant) and node.value in BENCH_ONLY), path.name
            # A command line of mlredact's: a list of arguments that starts with one of its commands.
            if isinstance(node, ast.List) and node.elts and _name(node.elts[0]) in COMMANDS:
                started.append((path.name, _name(node.elts[0])))
    assert {"bench", "bench-version"} <= COMMANDS  # the benchmark's commands are among them
    # The job's command line (api/runner.py ``_args``), and the system check's.
    assert started == [("app.py", "selfcheck"), ("runner.py", "run")]


# ---------------------------------------------------------------------------------------- a production job
@dataclass
class Watched:
    """What ran while :func:`_watch` was open: the source files of the functions started, and every module an import
    statement or ``importlib.import_module`` asked for (with each name it imports from it)."""

    files: set[str] = field(default_factory=set)
    imports: list[str] = field(default_factory=list)


@contextlib.contextmanager
def _watch() -> Iterator[Watched]:
    monitoring = sys.monitoring
    tool = next((t for t in range(6) if monitoring.get_tool(t) is None), None)
    if tool is None:
        pytest.skip("every monitoring tool id is taken (a debugger or a coverage tool is running)")
    watched = Watched()
    real_import, real_import_module = builtins.__import__, importlib.import_module

    def started(code: CodeType, _offset: int) -> Any:
        watched.files.add(code.co_filename)
        return monitoring.DISABLE  # once per function is enough

    def recorded_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        module = importlib.util.resolve_name("." * level + name, (globals or {}).get("__package__")) if level else name
        watched.imports += [module, *(f"{module}.{n}" for n in fromlist or ())]
        return real_import(name, globals, locals, fromlist, level)

    def recorded_import_module(name: str, package: str | None = None) -> Any:
        watched.imports.append(importlib.util.resolve_name(name, package) if name.startswith(".") else name)
        return real_import_module(name, package)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", recorded_import)
        mp.setattr(importlib, "import_module", recorded_import_module)
        monitoring.use_tool_id(tool, "production-paths")
        monitoring.register_callback(tool, monitoring.events.PY_START, started)
        monitoring.restart_events()
        monitoring.set_events(tool, monitoring.events.PY_START)
        try:
            yield watched
        finally:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool)


def test_a_production_job_never_imports_or_runs_the_benchmark_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mlredact run`` on the letter, while every function started and every import is recorded."""
    pool = InProcessPool()
    pool.install(monkeypatch)
    monkeypatch.setenv("MLREDACT_MODELS_DIR", str(tmp_path))  # never read: model files are not resolved
    config, src, out = tmp_path / "no-models.yaml", tmp_path / "input.pdf", tmp_path / "out"
    config.write_text(yaml.safe_dump(NO_MODELS), "utf-8")
    src.write_bytes(LETTER)
    with _watch() as watched:
        res = CliRunner().invoke(app, ["run", str(src), "--out", str(out), "--config", str(config), "--dev-key"])
    assert res.exit_code == 0, res.output
    assert (out / "redacted.pdf").read_bytes() == pool.builds[-1].pdf
    # What ran: the command, the runner and the worker tasks, ...
    ran = {Path(f).resolve() for f in watched.files}
    assert {SRC / "cli" / "main.py", SRC / "pipeline" / "runner.py", SRC / "pipeline" / "workers.py"} <= ran
    assert "mlredact.pipeline.runner.Redactor" in watched.imports  # the command imports the runner as it runs
    # ... and nothing of the adapter's.
    assert [f for f in sorted(watched.files) if _adapter_file(f)] == []
    assert [i for i in watched.imports if _adapter_module(i)] == []
