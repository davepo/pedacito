"""Spec-driven design: interfaces, data shapes, and tests before code.

    design = generate(client, cfg, spec, root)
    warnings = validate(design, cfg)
    print(render(design))

The design is a small JSON contract produced by the model, then saved under
`.pedacito/design.json` and `.pedacito/design.md`. The planner and interface
step get its compact form, and `new --design` builds test stubs before source
files. Validation is deterministic: duplicate paths, missing dependencies,
cycles, and over-wide modules are reported before any file is written.
"""

from __future__ import annotations

import json
from pathlib import Path

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string", "maxLength": 80},
        "summary": {"type": "string", "maxLength": 1200},
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 200},
                    "responsibility": {"type": "string", "maxLength": 400},
                    "interfaces": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 240},
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 200},
                    },
                    "estimated_lines": {"type": "integer"},
                },
                "required": ["path", "responsibility", "interfaces",
                             "depends_on"],
                "additionalProperties": False,
            },
        },
        "data_shapes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 120},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 160},
                    },
                },
                "required": ["name", "fields"],
                "additionalProperties": False,
            },
        },
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 200},
                    "covers": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 200},
                    },
                    "cases": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 240},
                    },
                },
                "required": ["path", "covers", "cases"],
                "additionalProperties": False,
            },
        },
        "risks": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
        },
    },
    "required": ["project_name", "summary", "modules", "data_shapes",
                 "tests", "risks"],
    "additionalProperties": False,
}

DESIGN_SYSTEM = """\
You are a software architect. Before any code is written, produce a compact
design for the requested project.

Rules:
- One module per file. Give each a single responsibility and its public
  interfaces (function/class signatures or one-line contracts), not bodies.
- List data shapes with their fields; list tests with the modules they cover
  and concrete cases.
- Dependencies must name files in the module list. Keep the graph acyclic.
- Keep modules small enough to implement in one focused pass; split anything
  that would exceed roughly 200 lines.
- Do not invent third-party packages. Prefer the standard library.
"""


def _path(value: str) -> str:
    p = (value or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _norm(design: dict) -> dict:
    """Copy a design with normalized paths and deduplicated string lists."""
    out = {
        "project_name": str(design.get("project_name") or "project")[:80],
        "summary": str(design.get("summary") or "")[:1200],
        "modules": [],
        "data_shapes": [],
        "tests": [],
        "risks": [str(r)[:300] for r in (design.get("risks") or [])
                  if str(r).strip()],
    }
    for m in design.get("modules") or []:
        if not isinstance(m, dict):
            continue
        path = _path(str(m.get("path") or ""))
        if not path:
            continue
        try:
            estimated = max(0, int(m.get("estimated_lines") or 0))
        except (TypeError, ValueError):
            estimated = 0
        out["modules"].append({
            "path": path,
            "responsibility": str(m.get("responsibility") or "")[:400],
            "interfaces": [str(x)[:240] for x in (m.get("interfaces") or [])
                           if str(x).strip()][:30],
            "depends_on": [_path(str(x)) for x in (m.get("depends_on") or [])
                           if _path(str(x))][:20],
            "estimated_lines": estimated,
        })
    for d in design.get("data_shapes") or []:
        if isinstance(d, dict) and str(d.get("name") or "").strip():
            out["data_shapes"].append({
                "name": str(d.get("name"))[:120],
                "fields": [str(x)[:160] for x in (d.get("fields") or [])
                           if str(x).strip()][:30],
            })
    for t in design.get("tests") or []:
        if not isinstance(t, dict):
            continue
        path = _path(str(t.get("path") or ""))
        if path:
            out["tests"].append({
                "path": path,
                "covers": [_path(str(x)) for x in (t.get("covers") or [])
                           if _path(str(x))][:20],
                "cases": [str(x)[:240] for x in (t.get("cases") or [])
                          if str(x).strip()][:30],
            })
    return out


def validate(design: dict, cfg=None) -> list[str]:
    """Deterministic design warnings: duplicates, bad deps, cycles, width."""
    design = _norm(design)
    warnings: list[str] = []
    paths = [m["path"] for m in design["modules"]]
    seen = set()
    for p in paths:
        if p in seen:
            warnings.append(f"duplicate module path: {p}")
        seen.add(p)
    graph = {m["path"]: list(m["depends_on"]) for m in design["modules"]}
    for m in design["modules"]:
        for dep in m["depends_on"]:
            if dep not in graph:
                warnings.append(f"{m['path']} depends on unknown module {dep}")
            if dep == m["path"]:
                warnings.append(f"{m['path']} depends on itself")
    # DFS cycle report (one example per SCC is enough).
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str):
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                continue
            if state.get(dep) == 1:
                i = stack.index(dep)
                warnings.append("dependency cycle: " + " -> ".join(
                    stack[i:] + [dep]))
            elif state.get(dep, 0) == 0:
                visit(dep)
        stack.pop()
        state[node] = 2

    for p in graph:
        if state.get(p, 0) == 0:
            visit(p)
    max_lines = int(getattr(cfg, "auto_split_lines", 120) or 120)
    for m in design["modules"]:
        if m["estimated_lines"] > max_lines:
            warnings.append(
                f"{m['path']} is estimated at {m['estimated_lines']} lines "
                f"(> auto_split_lines={max_lines}); split it")
        if len(m["interfaces"]) > 12:
            warnings.append(f"{m['path']} exposes {len(m['interfaces'])} "
                            "interfaces; split the module")
    if design["modules"] and not design["tests"]:
        warnings.append("no tests are designed for the modules")
    return warnings


def render(design: dict) -> str:
    """Markdown design document saved under .pedacito/design.md."""
    d = _norm(design)
    lines = [f"# Design: {d['project_name']}", "", d["summary"], "",
             "## Modules", ""]
    for m in d["modules"]:
        lines.append(f"### `{m['path']}`")
        lines.append(m["responsibility"] or "(no responsibility stated)")
        if m["interfaces"]:
            lines += ["", "Interfaces:"] + [f"- `{x}`" for x in m["interfaces"]]
        if m["depends_on"]:
            lines += ["", "Depends on: " + ", ".join(
                f"`{x}`" for x in m["depends_on"])]
        if m["estimated_lines"]:
            lines += ["", f"Estimated lines: {m['estimated_lines']}"]
        lines.append("")
    if d["data_shapes"]:
        lines += ["## Data shapes", ""]
        for s in d["data_shapes"]:
            lines.append(f"- **{s['name']}**: " + ", ".join(s["fields"]))
        lines.append("")
    if d["tests"]:
        lines += ["## Tests", ""]
        for t in d["tests"]:
            lines.append(f"- `{t['path']}` covers " +
                         (", ".join(f"`{x}`" for x in t["covers"]) or "?") +
                         (": " + "; ".join(t["cases"]) if t["cases"] else ""))
        lines.append("")
    if d["risks"]:
        lines += ["## Risks", ""] + [f"- {r}" for r in d["risks"]]
    return "\n".join(lines).rstrip() + "\n"


def planner_context(design: dict, limit: int = 2500) -> str:
    """Compact design summary for the file-planning prompt."""
    d = _norm(design)
    lines = [f"DESIGN ({d['project_name']}): {d['summary']}"]
    for m in d["modules"]:
        deps = f" <- {', '.join(m['depends_on'])}" if m["depends_on"] else ""
        lines.append(f"- {m['path']}: {m['responsibility']}{deps}")
    if d["tests"]:
        lines.append("tests: " + ", ".join(t["path"] for t in d["tests"]))
    text = "\n".join(lines)
    return text[:limit]


def load(root) -> dict | None:
    """Read `.pedacito/design.json`, or None when no design was saved."""
    from .index import state_dir
    try:
        raw = (state_dir(Path(root)) / "design.json").read_text(
            encoding="utf-8")
        data = json.loads(raw)
        return _norm(data) if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def brief(root, limit: int = 1800) -> str:
    """A short design note for task prompts, or '' when none was saved."""
    d = load(root)
    if not d:
        return ""
    return ("DESIGN BRIEF (from .pedacito/design.md; follow the interfaces "
            "and tests unless the task says otherwise):\n"
            + planner_context(d, limit=limit))


def save(root, design: dict) -> tuple[Path, Path]:
    """Write `.pedacito/design.json` and `.pedacito/design.md`."""
    from .index import state_dir
    d = state_dir(Path(root))
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "design.json"
    mp = d / "design.md"
    jp.write_text(json.dumps(_norm(design), indent=1), encoding="utf-8")
    mp.write_text(render(design), encoding="utf-8")
    return jp, mp


def generate(client, cfg, spec, root) -> dict:
    """One JSON design call, normalized and saved under `.pedacito/`."""
    messages = [
        {"role": "system", "content": DESIGN_SYSTEM},
        {"role": "user", "content":
         f"PROJECT SPECIFICATION\n\n{getattr(spec, 'text', spec)}"},
    ]
    raw = client.chat_json(
        messages, DESIGN_SCHEMA,
        max_tokens=int(getattr(cfg, "max_tokens_plan", 2000) or 2000),
        disable_thinking=True)
    design = _norm(raw if isinstance(raw, dict) else {})
    try:
        save(root, design)
    except OSError:
        pass                       # the design still guides this run
    return design
