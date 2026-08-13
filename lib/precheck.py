#!/usr/bin/env python3
"""Static name check for a Kaggle kernel — every unbound global, before the GPU sees it.

Two kernels in a row died on a NameError that this catches in 40 milliseconds: a variable used
at module scope that no assignment ever created. I ran this on one file, skipped it on the file
derived FROM that one, and lost a full run to `OUT` being undefined two lines after the model
finished loading. Cheap checks only help if they are unskippable, so push_verified calls this and
refuses to push when it fails.
"""
import ast, builtins, sys
from pathlib import Path


def unbound(path):
    tree = ast.parse(Path(path).read_text())
    bound = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
            bound.add(n.optional_vars.id)
        elif isinstance(n, (ast.comprehension, ast.For, ast.AsyncFor)):
            for e in ast.walk(n.target):
                if isinstance(e, ast.Name):
                    bound.add(e.id)
    used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    return sorted(u for u in used - bound if not u.startswith("_"))


if __name__ == "__main__":
    bad = unbound(sys.argv[1])
    print(f"{sys.argv[1]}: " + (f"UNBOUND {bad}" if bad else "clean"))
    sys.exit(1 if bad else 0)
