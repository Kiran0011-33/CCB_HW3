from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple
import math

# ------------------------------------------------------------
# Optional image generation support
# Requires:
#   pip install graphviz
# and Graphviz installed on your machine
# ------------------------------------------------------------
try:
    from graphviz import Digraph
    GRAPHVIZ_OK = True
except Exception:
    GRAPHVIZ_OK = False


# ------------------------------------------------------------
# Expression tree
# ------------------------------------------------------------
@dataclass(frozen=True)
class Expr:
    op: str
    left: "Expr | None" = None
    right: "Expr | None" = None
    name: str | None = None
    prob: Fraction | None = None

    def value(self) -> Fraction:
        if self.op == "SRC":
            assert self.prob is not None
            return self.prob
        if self.op == "CONST0":
            return Fraction(0, 1)
        if self.op == "CONST1":
            return Fraction(1, 1)
        if self.op == "NOT":
            assert self.left is not None
            return Fraction(1, 1) - self.left.value()
        if self.op == "AND":
            assert self.left is not None and self.right is not None
            return self.left.value() * self.right.value()
        raise ValueError(f"Unknown op: {self.op}")

    def size(self) -> int:
        if self.op in {"SRC", "CONST0", "CONST1"}:
            return 1
        if self.op == "NOT":
            assert self.left is not None
            return 1 + self.left.size()
        if self.op == "AND":
            assert self.left is not None and self.right is not None
            return 1 + self.left.size() + self.right.size()
        raise ValueError(f"Unknown op: {self.op}")

    def to_text(self) -> str:
        if self.op == "SRC":
            assert self.name is not None
            return self.name
        if self.op == "CONST0":
            return "0"
        if self.op == "CONST1":
            return "1"
        if self.op == "NOT":
            assert self.left is not None
            return f"NOT({self.left.to_text()})"
        if self.op == "AND":
            assert self.left is not None and self.right is not None
            return f"AND({self.left.to_text()}, {self.right.to_text()})"
        raise ValueError(f"Unknown op: {self.op}")

    def canonical(self) -> str:
        if self.op == "SRC":
            assert self.name is not None
            return self.name
        if self.op == "CONST0":
            return "0"
        if self.op == "CONST1":
            return "1"
        if self.op == "NOT":
            assert self.left is not None
            return f"N({self.left.canonical()})"
        if self.op == "AND":
            assert self.left is not None and self.right is not None
            a = self.left.canonical()
            b = self.right.canonical()
            if a <= b:
                return f"A({a},{b})"
            return f"A({b},{a})"
        raise ValueError(f"Unknown op: {self.op}")


# ------------------------------------------------------------
# Constructors
# ------------------------------------------------------------
def SRC(name: str, p: Fraction) -> Expr:
    return Expr("SRC", name=name, prob=p)

def CONST0() -> Expr:
    return Expr("CONST0")

def CONST1() -> Expr:
    return Expr("CONST1")

def NOT(x: Expr) -> Expr:
    return Expr("NOT", left=x)

def AND(x: Expr, y: Expr) -> Expr:
    # Canonicalize commutative order for easier dedup
    if x.canonical() <= y.canonical():
        return Expr("AND", left=x, right=y)
    return Expr("AND", left=y, right=x)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def frac_from_decimal_string(s: str) -> Fraction:
    return Fraction(Decimal(s))

def score_expr(expr: Expr, target: Fraction) -> Tuple[Fraction, int, int]:
    # smaller is better
    diff = abs(expr.value() - target)
    return (diff, expr.size(), len(expr.to_text()))

def best_by_value(candidates: List[Expr]) -> Dict[Fraction, Expr]:
    best: Dict[Fraction, Expr] = {}
    for e in candidates:
        v = e.value()
        if v not in best:
            best[v] = e
        else:
            old = best[v]
            if (e.size(), len(e.to_text())) < (old.size(), len(old.to_text())):
                best[v] = e
    return best


# ------------------------------------------------------------
# Question 2(a): beam search using only AND / NOT from {0.4, 0.5}
# ------------------------------------------------------------
def search_and_not_expression(
    target_str: str,
    max_rounds: int = 7,
    beam_width: int = 250,
    pair_limit: int = 180,
) -> Expr:
    """
    Heuristic beam search.
    - Returns exact expression if found in the explored space.
    - Otherwise returns the nearest expression found.
    """

    target = frac_from_decimal_string(target_str)

    A = SRC("A", Fraction(2, 5))   # 0.4
    B = SRC("B", Fraction(1, 2))   # 0.5

    seeds = [A, B, NOT(A), NOT(B)]
    known = best_by_value(seeds)
    beam = list(known.values())

    best_expr = min(beam, key=lambda e: score_expr(e, target))
    if best_expr.value() == target:
        return best_expr

    for _ in range(max_rounds):
        beam = sorted(beam, key=lambda e: score_expr(e, target))[:beam_width]
        pool = beam[:pair_limit]

        new_candidates: List[Expr] = []
        # NOT expansions
        for e in pool:
            new_candidates.append(NOT(e))

        # AND expansions
        for i in range(len(pool)):
            for j in range(i, len(pool)):
                new_candidates.append(AND(pool[i], pool[j]))

        # Merge with old best-known expressions
        merged = list(known.values()) + new_candidates
        known = best_by_value(merged)

        beam = sorted(known.values(), key=lambda e: score_expr(e, target))[:beam_width]

        current_best = beam[0]
        if score_expr(current_best, target) < score_expr(best_expr, target):
            best_expr = current_best

        if current_best.value() == target:
            return current_best

    return best_expr


# ------------------------------------------------------------
# Question 2(b): exact binary construction from {0.5}
#
# If q is the suffix value and H = 0.5:
# bit 0: new = q/2 = AND(H, q)
# bit 1: new = 1 - (1-q)/2 = NOT(AND(H, NOT(q)))
#
# This uses AND / NOT logic and a constant 0 as the base suffix.
# ------------------------------------------------------------
def build_binary_expression(bits: str) -> Expr:
    H = SRC("H", Fraction(1, 2))  # 0.5 source
    expr = CONST0()

    for bit in reversed(bits):
        if bit == "0":
            expr = AND(H, expr)
        elif bit == "1":
            expr = NOT(AND(H, NOT(expr)))
        else:
            raise ValueError("bits must contain only 0 or 1")
    return expr


# ------------------------------------------------------------
# Diagram generation
# ------------------------------------------------------------
def render_expr_png(expr: Expr, out_png_without_ext: str) -> None:
    if not GRAPHVIZ_OK:
        print(f"[warning] graphviz python package not available; skipped image for {out_png_without_ext}")
        return

    dot = Digraph(format="png")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded")

    node_count = {"n": 0}

    def add_node(e: Expr) -> str:
        node_count["n"] += 1
        node_id = f"n{node_count['n']}"

        if e.op == "SRC":
            label = f"{e.name}\\nP={float(e.value()):.7f}"
            dot.node(node_id, label)
            return node_id

        if e.op == "CONST0":
            dot.node(node_id, "CONST 0\\nP=0")
            return node_id

        if e.op == "CONST1":
            dot.node(node_id, "CONST 1\\nP=1")
            return node_id

        if e.op == "NOT":
            dot.node(node_id, f"NOT\\nP={float(e.value()):.7f}")
            c = add_node(e.left)
            dot.edge(c, node_id)
            return node_id

        if e.op == "AND":
            dot.node(node_id, f"AND\\nP={float(e.value()):.7f}")
            l = add_node(e.left)
            r = add_node(e.right)
            dot.edge(l, node_id)
            dot.edge(r, node_id)
            return node_id

        raise ValueError(f"Unknown op: {e.op}")

    root_id = add_node(expr)
    dot.node("OUT", f"OUTPUT\\nP={float(expr.value()):.7f}", shape="ellipse")
    dot.edge(root_id, "OUT")
    dot.render(out_png_without_ext, cleanup=True)


# ------------------------------------------------------------
# Pretty reporting
# ------------------------------------------------------------
def print_solution_block(title: str, expr: Expr, target: Fraction | None = None) -> None:
    print("=" * 80)
    print(title)
    print("- Expression:")
    print(expr.to_text())
    print("- Value:")
    print(f"  exact = {expr.value()}")
    print(f"  float = {float(expr.value()):.10f}")
    print(f"- Size: {expr.size()}")
    if target is not None:
        err = abs(expr.value() - target)
        print(f"- Target:")
        print(f"  exact = {target}")
        print(f"  float = {float(target):.10f}")
        print(f"- Absolute error:")
        print(f"  exact = {err}")
        print(f"  float = {float(err):.10f}")
    print()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    out_dir = Path("q2_outputs")
    out_dir.mkdir(exist_ok=True)

    # -------------------------
    # Question 2(a)
    # -------------------------
    print("\nQUESTION 2(a): search from S = {0.4, 0.5} using AND / NOT\n")

    targets_2a = ["0.8881188", "0.2119209", "0.5555555"]

    for idx, t in enumerate(targets_2a, start=1):
        expr = search_and_not_expression(
            target_str=t,
            max_rounds=7,      # increase to 8 or 9 if you want a deeper search
            beam_width=250,
            pair_limit=180,
        )
        target_frac = frac_from_decimal_string(t)

        print_solution_block(f"Q2(a).{idx} target = {t}", expr, target_frac)
        render_expr_png(expr, str(out_dir / f"q2a_{idx}"))

    # -------------------------
    # Question 2(b)
    # -------------------------
    print("\nQUESTION 2(b): exact binary construction from S = {0.5}\n")

    binaries = ["1011111", "1101111", "1010111"]

    for idx, bits in enumerate(binaries, start=1):
        expr = build_binary_expression(bits)

        # exact target from binary string
        target = Fraction(0, 1)
        for i, b in enumerate(bits, start=1):
            if b == "1":
                target += Fraction(1, 2 ** i)

        print_solution_block(f"Q2(b).{idx} target = 0.{bits}_2", expr, target)
        render_expr_png(expr, str(out_dir / f"q2b_{idx}"))

    print(f"Done. Check the folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
