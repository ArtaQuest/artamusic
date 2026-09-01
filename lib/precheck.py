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


def _bound_here(node, include_nested_names=True):
    """Names bound directly in this scope: assignments, imports, defs, params, except/with/for."""
    import ast as _a
    out = set()
    if isinstance(node, (_a.FunctionDef, _a.AsyncFunctionDef)):
        a = node.args
        out |= {x.arg for x in a.args + a.kwonlyargs + getattr(a, "posonlyargs", [])}
        if a.vararg: out.add(a.vararg.arg)
        if a.kwarg: out.add(a.kwarg.arg)
        body = node.body
    else:
        body = node.body
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (_a.FunctionDef, _a.AsyncFunctionDef, _a.ClassDef)):
            out.add(n.name)                       # do NOT descend: its body is its own scope
            continue
        if isinstance(n, (_a.Import, _a.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
            continue
        if isinstance(n, _a.Lambda):
            continue
        if isinstance(n, _a.Name) and isinstance(n.ctx, (_a.Store, _a.Del)):
            out.add(n.id)
        elif isinstance(n, _a.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (_a.Global, _a.Nonlocal)):
            out.update(n.names)
        stack.extend(_a.iter_child_nodes(n))
    return out


def _scope_check(node, enclosing, problems, label):
    """Report Name loads in THIS scope that no enclosing scope binds; recurse into nested defs."""
    import ast as _a
    here = _bound_here(node) | enclosing
    stack = list(node.body)
    nested = []
    while stack:
        n = stack.pop()
        if isinstance(n, (_a.FunctionDef, _a.AsyncFunctionDef)):
            nested.append(n); continue
        if isinstance(n, _a.ClassDef):
            nested.append(n); continue
        if isinstance(n, _a.Lambda):
            # a lambda is its own scope: its parameters bind inside it, and descending into its
            # body without them flagged every `key=lambda L: L` in the first file that used one.
            continue
        if isinstance(n, _a.Name) and isinstance(n.ctx, _a.Load) and n.id not in here:
            problems.append(f"{label}: {n.id}")
        stack.extend(_a.iter_child_nodes(n))
    for fn in nested:
        _scope_check(fn, here, problems, f"{label}.{fn.name}" if label else fn.name)


def unbound_in_functions(path_or_src, is_src=False):
    """Names a FUNCTION reads that nothing binds — module scope, an enclosing scope, its own
    locals, or builtins.

    `unbound()` above walks module scope only, and that is where every death it was written for
    lived. Then the record notebook grew stage functions run in their own process, an import that
    served two of them stayed behind in one, and the second stage died on the GPU at second three
    with NameError: snapshot_download. A function body is module scope's blind spot; this closes
    it — with a real scope chain, because a checker that flags every closure is one nobody reads.
    """
    import builtins
    src = path_or_src if is_src else Path(path_or_src).read_text()
    tree = ast.parse(src)
    gl = set(dir(builtins)) | {"__name__", "__file__", "__doc__"} | _bound_here(tree)
    problems = []
    for fn in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        _scope_check(fn, gl, problems, fn.name)
    return sorted(set(problems))

def used_before_defined(path_or_src, is_src=False):
    """Module-scope names USED above the line that binds them.

    `unbound()` asks whether a name is bound anywhere in the module, which is not the question
    Python answers at run time. A stage source whose `if __name__ == "__main__":` dispatcher sat
    ABOVE the function it dispatches to passed every check here and then died on the GPU with
    `NameError: name 'stage_cover' is not defined` — the name was bound, just not yet. Same shape
    as reading a model's device map after deleting the model: static binding is not ordering.

    Two things it must NOT flag, both learned by flagging them:
      * a name referenced inside a function body, which does not run where it is written;
      * a name bound inside a `for`/`if`/`while`/`try` at module scope — those bindings are real,
        and missing them made this report seventeen false positives on its first outing. A check
        that cries wolf is worse than no check, because it gets skipped.
    """
    import ast as _a
    from pathlib import Path as _P
    src = path_or_src if is_src else _P(path_or_src).read_text()
    tree = _a.parse(src)
    SCOPED = (_a.FunctionDef, _a.AsyncFunctionDef, _a.ClassDef, _a.Lambda)

    def walk_module(node):
        """Every node that EXECUTES at module scope — descending into compound statements but
        never into a function or class body."""
        for child in _a.iter_child_nodes(node):
            if isinstance(child, SCOPED):
                yield child                      # the def itself binds a name; its body does not run
                continue
            yield child
            yield from walk_module(child)

    bound_at = {}
    def note(name, line):
        if name not in bound_at or line < bound_at[name]:
            bound_at[name] = line

    for n in walk_module(tree):
        if isinstance(n, SCOPED) and hasattr(n, "name"):
            note(n.name, n.lineno)
        elif isinstance(n, (_a.Assign, _a.AugAssign, _a.AnnAssign, _a.For, _a.AsyncFor)):
            tgts = n.targets if isinstance(n, _a.Assign) else [getattr(n, "target", None)]
            for t in filter(None, tgts):
                for x in _a.walk(t):
                    if isinstance(x, _a.Name):
                        note(x.id, n.lineno)
        elif isinstance(n, (_a.Import, _a.ImportFrom)):
            for al in n.names:
                note((al.asname or al.name).split(".")[0], n.lineno)
        elif isinstance(n, (_a.With, _a.AsyncWith)):
            for item in n.items:
                for x in _a.walk(item.optional_vars) if item.optional_vars else []:
                    if isinstance(x, _a.Name):
                        note(x.id, n.lineno)
        elif isinstance(n, _a.ExceptHandler) and n.name:
            note(n.name, n.lineno)
        elif isinstance(n, (_a.DictComp, _a.ListComp, _a.SetComp, _a.GeneratorExp)):
            # A comprehension target is bound for the WHOLE expression, whatever the layout: in
            #   {ax: f(ax)
            #    for ax in axes}
            # the key expression sits a line ABOVE the `for` that binds it, and noting the target
            # at its own line flagged exactly that as used-before-defined. Note it at the line the
            # expression STARTS, which no use inside the expression can precede.
            for gen in n.generators:
                for x in _a.walk(gen.target):
                    if isinstance(x, _a.Name):
                        note(x.id, n.lineno)

    bad = []
    for n in walk_module(tree):
        if isinstance(n, SCOPED):
            continue
        if isinstance(n, _a.Name) and isinstance(n.ctx, _a.Load):
            at = bound_at.get(n.id)
            if at is not None and at > n.lineno:
                bad.append(f"{n.id} used at line {n.lineno}, bound at {at}")
    return sorted(set(bad))
