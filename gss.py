from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T", bound=float)

# 1 / phi  (inverse golden ratio)
INVPHI = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class GSSStep:
    """Snapshot of one golden section search iteration."""

    iteration: int
    a: float
    b: float
    c: float | None
    d: float | None
    fc: float | None
    fd: float | None
    keep_left: bool | None
    done: bool


def gss(
    f: Callable[[T], float],
    a: float,
    b: float,
    tolerance: float = 1e-5,
) -> float:
    """Return an approximate minimizer of unimodal ``f`` on ``[a, b]``.

    This implementation does not reuse function evaluations.

    Example
    -------
    >>> gss(lambda x: (x - 2) ** 2, 1, 5)
    2.0
    """
    while b - a > tolerance:
        c = b - (b - a) * INVPHI
        d = a + (b - a) * INVPHI
        if f(c) < f(d):
            b = d
        else:
            a = c
    return (b + a) / 2


def gss_iterate(
    f: Callable[[T], float],
    a: float,
    b: float,
    tolerance: float = 1e-5,
) -> Iterator[GSSStep]:
    """Yield iteration snapshots for visualization."""
    iteration = 0
    yield GSSStep(iteration, a, b, None, None, None, None, None, done=False)

    while b - a > tolerance:
        c = b - (b - a) * INVPHI
        d = a + (b - a) * INVPHI
        fc = f(c)
        fd = f(d)
        keep_left = fc < fd
        yield GSSStep(iteration, a, b, c, d, fc, fd, keep_left, done=False)

        if keep_left:
            b = d
        else:
            a = c
        iteration += 1

    yield GSSStep(iteration, a, b, None, None, None, None, None, done=True)
