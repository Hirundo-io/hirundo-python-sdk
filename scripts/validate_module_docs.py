#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_public_name(name: str) -> bool:
    if name.startswith("_"):
        return False
    if name.startswith("__") and name.endswith("__"):
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


def resolve_module_name(module_path: Path) -> str | None:
    stem = module_path.stem
    if stem == "__init__":
        return "hirundo"
    if stem.startswith("_") or stem == "__main__":
        return None
    return f"hirundo.{stem}"


def collect_module_exports(package_dir: Path) -> dict[str, set[str]]:
    module_exports: dict[str, set[str]] = {}
    for module_path in package_dir.glob("*.py"):
        module_name = resolve_module_name(module_path)
        if not module_name:
            continue
        exports = get_public_exports(module_path)
        if exports:
            module_exports[module_name] = exports
    return module_exports


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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "hirundo"
    docs_dir = repo_root / "docs"

    module_exports = collect_module_exports(package_dir)
    docs_modules = collect_docs_modules(docs_dir)

    public_modules = set(module_exports.keys())

    missing_docs = sorted(public_modules - docs_modules)
    extra_docs = sorted(docs_modules - public_modules)

    return report_mismatches(missing_docs, extra_docs)


if __name__ == "__main__":
    raise SystemExit(main())
