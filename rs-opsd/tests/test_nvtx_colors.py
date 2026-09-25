import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def test_extra_colors_do_not_require_matplotlib():
    # Test this small adapter without requiring GPU-only profiler packages on
    # the submit host. The PPU runtime preflight exercises real NVTX as well.
    path = Path(__file__).resolve().parents[1] / "verl/utils/profiler/nvtx_profile.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if
                 isinstance(node, ast.FunctionDef) and node.name in ("_portable_nvtx_color", "mark_start_range")
                 or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_EXTRA_NVTX_COLORS" for t in node.targets)]
    seen = []
    namespace = {"Optional": Optional, "nvtx": SimpleNamespace(start_range=lambda **kw: seen.append(kw["color"]))}
    exec(compile(tree, str(path), "exec"), namespace)
    for color in ("olive", "brown", "pink", "red", None):
        namespace["mark_start_range"](color=color)
    assert seen == [0x808000, 0xA52A2A, 0xFFC0CB, "red", None]
