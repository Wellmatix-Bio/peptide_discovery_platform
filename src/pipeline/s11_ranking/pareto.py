# Pareto fronts for Stage 11: candidates are layered by non-domination before the weighted score orders each layer.
from __future__ import annotations


def pareto_fronts(vectors: list[list[float]]) -> list[int]:
    """1-based front per vector (1 = non-dominated); higher is better in every column.

    A dominates B when A is at least as good in every column and strictly better in one.
    """
    n = len(vectors)
    dominated_by = [0] * n
    dominates: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vectors[i], vectors[j]
            if all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b)):
                dominates[i].append(j)
                dominated_by[j] += 1
            elif all(y >= x for x, y in zip(a, b)) and any(y > x for x, y in zip(a, b)):
                dominates[j].append(i)
                dominated_by[i] += 1

    fronts = [0] * n
    current = [i for i in range(n) if dominated_by[i] == 0]
    front = 1
    while current:
        following = []
        for i in current:
            fronts[i] = front
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    following.append(j)
        current, front = following, front + 1
    return fronts
