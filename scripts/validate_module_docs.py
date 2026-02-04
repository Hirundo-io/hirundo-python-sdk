#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_public_name(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.startswith("_"):
        return False
    return True


def extract_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for elt in target.elts:
            names.update(extract_target_names(elt))
        return names
    return set()


def extract_all_names(value: ast.AST | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, ast.List | ast.Tuple):
        names = []
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
        return set(names)
    return set()


def handle_definition(node: ast.AST, public_exports: set[str]) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        if is_public_name(node.name):
            public_exports.add(node.name)
        return True
    return False


def handle_assignment(
    targets: Sequence[ast.AST],
    value: ast.AST | None,
    public_exports: set[str],
    all_names: set[str],
) -> None:
    for target in targets:
        for name in extract_target_names(target):
            if name == "__all__":
                all_names.update(extract_all_names(value))
            elif is_public_name(name):
                public_exports.add(name)


def collect_exports(nodes: Sequence[ast.AST]) -> set[str]:
    public_exports: set[str] = set()
    all_names: set[str] = set()
    for node in nodes:
        if handle_definition(node, public_exports):
            continue
        if isinstance(node, ast.Assign):
            handle_assignment(node.targets, node.value, public_exports, all_names)
            continue
        if isinstance(node, ast.AnnAssign):
            handle_assignment([node.target], node.value, public_exports, all_names)
    return public_exports | all_names


def get_public_exports(module_path: Path) -> set[str]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    return collect_exports(tree.body)


def resolve_module_name(module_path: Path, package_dir: Path) -> str | None:
    relative = module_path.relative_to(package_dir)
    parts = relative.with_suffix("").parts
    if "__pycache__" in parts:
        return None
    if parts[-1] == "__main__":
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if any(part.startswith("_") for part in parts):
        return None
    if not parts:
        return "hirundo"
    return "hirundo." + ".".join(parts)


def collect_module_exports(package_dir: Path) -> tuple[dict[str, set[str]], list[Path]]:
    module_exports: dict[str, set[str]] = {}
    parse_errors: list[Path] = []
    for module_path in package_dir.rglob("*.py"):
        module_name = resolve_module_name(module_path, package_dir)
        if not module_name:
            continue
        try:
            exports = get_public_exports(module_path)
        except SyntaxError:
            parse_errors.append(module_path)
            continue
        if exports:
            module_exports[module_name] = exports
    return module_exports, parse_errors


def collect_docs_modules(docs_dir: Path) -> set[str]:
    docs_modules: set[str] = set()
    for doc_path in docs_dir.glob("hirundo*.rst"):
        stem = doc_path.stem
        if stem == "hirundo":
            docs_modules.add("hirundo")
        elif stem.startswith("hirundo."):
            docs_modules.add(stem)
    return docs_modules


def report_mismatches(missing_docs: list[str], extra_docs: list[str]) -> int:
    if not missing_docs and not extra_docs:
        print("Module docs check passed.")
        return 0
    if missing_docs:
        print("Missing .rst for public modules:")
        for module in missing_docs:
            print(f"  - {module}")
    if extra_docs:
        print(".rst files without public module exports:")
        for module in extra_docs:
            print(f"  - {module}")
    return 1


def report_parse_errors(parse_errors: list[Path]) -> int:
    if not parse_errors:
        return 0
    print("Failed to parse module source files:")
    for module_path in parse_errors:
        print(f"  - {module_path}")
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "hirundo"
    docs_dir = repo_root / "docs"

    module_exports, parse_errors = collect_module_exports(package_dir)
    docs_modules = collect_docs_modules(docs_dir)

    public_modules = set(module_exports.keys())

    missing_docs = sorted(public_modules - docs_modules)
    extra_docs = sorted(docs_modules - public_modules)

    parse_status = report_parse_errors(parse_errors)
    mismatch_status = report_mismatches(missing_docs, extra_docs)
    return 1 if parse_status or mismatch_status else 0


if __name__ == "__main__":
    raise SystemExit(main())
