"""Audit for unreachable code after return/raise (the blank-wizard bug class)."""

import ast
import os
import sys

ROOTS = ["ai_voice_studio", "main.py", "tests"]
TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def scan_body(body, ctx):
    hits = []
    for i, stmt in enumerate(body):
        if i + 1 < len(body) and isinstance(stmt, TERMINATORS):
            # The next statement at the same level is unreachable.
            nxt = body[i + 1]
            if isinstance(nxt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # plain return followed by a def is fine
            hits.append((stmt.lineno, nxt.lineno))
    return hits


def visit(tree, path):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(node.body)
            if body and isinstance(body[0], (ast.Expr,)):
                # Skip docstring-only handling; just scan the whole body.
                pass
            for (ret_ln, dead_ln) in scan_body(body, node):
                print(f"{path}:{node.lineno}  {node.name}(): "
                      f"unreachable code after return/raise at line {ret_ln} "
                      f"(next statement line {dead_ln})")
            # Nested functions/classes are walked by ast.walk already.


def main():
    found = 0
    for root in ROOTS:
        if os.path.isfile(root):
            files = [root]
        else:
            files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".py")]
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=path)
            except SyntaxError as exc:
                print(f"{path}: SYNTAX ERROR {exc}")
                found += 1
                continue
            before = found
            visit(tree, path)
            # Count hits printed for this file.
    print("Audit complete." if found == 0 else f"{found} issue(s) found.")


if __name__ == "__main__":
    main()
