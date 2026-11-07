"""Undeclared-import dependency audit (transferred pattern, 2026-11-07).

Every direct third-party import in runtime code must be declared in
pyproject.toml [project].dependencies. Undeclared imports work only via
transitive dependencies (e.g. dateutil via pandas) and break clean installs.

Technique source: hyperliquid-dex incident -> transfer opportunity (goal_generator).
Metric: count of undeclared direct third-party imports in runtime code (target: 0).
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# import-name -> pip-name where simple normalization is not enough
ALIASES = {"dateutil": "python-dateutil", "dotenv": "python-dotenv"}

RUNTIME_DIRS = ["tradingagents", "ec2_deploy", "cli"]
DEV_TOOLS = {"pytest"}  # test-only imports, intentionally not runtime deps


def declared_dependencies():
    txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", txt, re.S)
    deps = set()
    for dep in re.findall(r'"([^"]+)"', m.group(1)):
        name = re.split(r"[><=!~\[]", dep)[0].strip().lower()
        deps.add(name.replace("-", "_"))
    return deps


def local_toplevel_names():
    names = set()
    for p in ROOT.iterdir():
        if p.is_dir() and p.name not in (".venv", ".git", "__pycache__", "tests",
                                         "node_modules", ".pytest_cache"):
            names.add(p.name)
        elif p.is_file() and p.suffix == ".py":
            names.add(p.stem)
    # sibling modules imported top-level from inside runtime dirs
    # (e.g. `import config` from ec2_deploy/daily_run.py)
    for d in RUNTIME_DIRS:
        base = ROOT / d
        if base.is_dir():
            for py in base.glob("*.py"):
                names.add(py.stem)
            for sub in base.iterdir():
                if sub.is_dir() and sub.name != "__pycache__":
                    names.add(sub.name)
    return names


def direct_thirdparty_imports():
    stdlib = set(sys.stdlib_module_names)
    local = local_toplevel_names()
    found = {}  # top-level import -> set of files
    for d in RUNTIME_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    mods = [node.module]
                for mod in mods:
                    top = mod.split(".")[0]
                    if top in stdlib or top in local or top in DEV_TOOLS:
                        continue
                    found.setdefault(top, set()).add(str(p.relative_to(ROOT)))
    return found


def test_no_undeclared_thirdparty_imports():
    declared = declared_dependencies()
    imports = direct_thirdparty_imports()
    undeclared = {}
    for top, files in imports.items():
        pip_name = ALIASES.get(top, top).replace("-", "_")
        if pip_name not in declared:
            undeclared[top] = sorted(files)
    assert not undeclared, (
        "Undeclared direct third-party imports (work only via transitive deps): "
        + "; ".join(f"{k} imported by {v}" for k, v in undeclared.items())
    )
